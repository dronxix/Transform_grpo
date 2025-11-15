import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# Проверяем доступность PyTorch SDPA (доступен в PyTorch 2.0+)
HAS_SDPA = hasattr(F, 'scaled_dot_product_attention')

@dataclass
class ModelConfig:
    """Конфигурация модели"""
    obs_dim: int = 1000
    action_dim: int = 5
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 4
    num_experts: int = 8
    expert_capacity: float = 1.25  # Для load balancing
    top_k_experts: int = 2
    max_seq_len: int = 512
    dropout: float = 0.1
    use_flash_attn: bool = True
    use_gradient_checkpointing: bool = False
    rope_base: int = 10000
    load_balancing_loss_coef: float = 0.01


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
        
    def forward(self, seq_len: int, device: torch.device):
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Применяет RoPE к query и key тензорам"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) с поддержкой:
    - Flash Attention (если установлен)
    - PyTorch SDPA (fallback для PyTorch 2.0+)
    - Ручная реализация (fallback для старых версий)
    - RoPE
    - KV-Cache
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.num_queries_per_kv = config.num_heads // config.num_kv_heads
        
        # Q проекции для всех heads
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        
        # K, V проекции для меньшего количества KV heads
        self.k_proj = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, config.num_kv_heads * self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.dropout_p = config.dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, 
            max_seq_len=config.max_seq_len,
            base=config.rope_base
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        
        # Q: [batch, seq_len, num_heads, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # K, V: [batch, seq_len, num_kv_heads, head_dim]
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose для attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Применяем RoPE
        cos, sin = self.rotary_emb(seq_len, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # KV-Cache для инференса
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_cache = (k, v) if use_cache else None
        
        # Повторяем KV для каждой группы Q heads
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # ===== Выбор метода attention =====
        # 1. Flash Attention (самый быстрый, если установлен)
        if HAS_FLASH_ATTN and self.training and mask is None:
            # Flash attention требует [batch, seq_len, num_heads, head_dim]
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            attn_output = flash_attn_func(
                q.half(), k.half(), v.half(),
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=True
            ).float()
            
        # 2. PyTorch SDPA (оптимизированный, PyTorch 2.0+)
        elif HAS_SDPA:
            # SDPA работает с [batch, num_heads, seq_len, head_dim]
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=(mask is None),  # Causal только если нет custom маски
                scale=None  # Автоматический scaling 1/sqrt(head_dim)
            )
            # Transpose обратно: [batch, seq_len, num_heads, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            
        # 3. Ручная реализация (fallback для старых версий PyTorch)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # [batch, num_heads, seq_len, head_dim]
            attn_output = torch.matmul(attn_weights, v)
            # [batch, seq_len, num_heads, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
        
        # [batch, seq_len, embed_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, new_cache

class MoELayer(nn.Module):
    """
    Улучшенный Mixture of Experts с:
    - Load balancing loss
    - Эффективной векторизованной реализацией
    - Top-k routing
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.embed_dim = config.embed_dim
        expert_dim = 4 * config.embed_dim
        
        # Router
        self.router = nn.Linear(config.embed_dim, config.num_experts, bias=False)
        
        # Все эксперты в одном Linear для параллелизации
        # Используем SwiGLU активацию (gate + up проекции)
        self.expert_gate = nn.Linear(config.embed_dim, config.num_experts * expert_dim, bias=False)
        self.expert_up = nn.Linear(config.embed_dim, config.num_experts * expert_dim, bias=False)
        self.expert_down = nn.Linear(config.num_experts * expert_dim, config.embed_dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Для подсчета load balancing loss
        self.register_buffer("expert_usage", torch.zeros(config.num_experts))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.reshape(-1, embed_dim)  # [batch*seq_len, embed_dim]
        
        # Router logits
        router_logits = self.router(x_flat)  # [batch*seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k experts selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Load balancing: считаем использование экспертов
        if self.training:
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
            expert_usage = expert_mask.sum(dim=[0, 1]) / (x_flat.shape[0] * self.top_k)
            self.expert_usage = 0.9 * self.expert_usage + 0.1 * expert_usage
        
        # Эффективная векторизованная обработка экспертами
        # SwiGLU: gate(x) * up(x) -> down
        gate_proj = F.silu(self.expert_gate(x_flat))  # [batch*seq, num_experts * expert_dim]
        up_proj = self.expert_up(x_flat)  # [batch*seq, num_experts * expert_dim]
        
        # Reshape для применения масок экспертов
        gate_proj = gate_proj.view(x_flat.shape[0], self.num_experts, -1)
        up_proj = up_proj.view(x_flat.shape[0], self.num_experts, -1)
        
        # Применяем routing: выбираем только top-k экспертов
        expert_outputs = []
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # [batch*seq]
            expert_weight = top_k_probs[:, i:i+1]  # [batch*seq, 1]
            
            # Выбираем выходы соответствующих экспертов
            batch_indices = torch.arange(x_flat.shape[0], device=x_flat.device)
            expert_gate = gate_proj[batch_indices, expert_idx]  # [batch*seq, expert_dim]
            expert_up = up_proj[batch_indices, expert_idx]
            
            expert_out = expert_gate * expert_up  # [batch*seq, expert_dim]
            expert_outputs.append(expert_weight.unsqueeze(-1) * expert_out)
        
        # Суммируем взвешенные выходы
        combined_expert_output = torch.stack(expert_outputs, dim=0).sum(dim=0)  # [batch*seq, expert_dim]
        
        # Final down projection
        output = self.expert_down(combined_expert_output.view(x_flat.shape[0], -1))
        output = self.dropout(output)
        
        # Load balancing loss
        lb_loss = self._load_balancing_loss(router_probs)
        
        return output.view(batch_size, seq_len, embed_dim), lb_loss
    
    def _load_balancing_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary loss для равномерного распределения нагрузки между экспертами
        """
        # Средняя вероятность каждого эксперта
        expert_probs = router_probs.mean(dim=0)
        
        # Количество токенов, обрабатываемых каждым экспертом
        # Идеально: каждый эксперт обрабатывает 1/num_experts токенов
        target = 1.0 / self.num_experts
        
        # Coefficient of variation loss
        loss = self.num_experts * torch.sum(expert_probs * torch.log(expert_probs / target + 1e-10))
        return loss


class TransformerBlock(nn.Module):
    """Transformer block с GQA и MoE"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.attention = GroupedQueryAttention(config)
        self.moe = MoELayer(config)
        
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.use_checkpoint = config.use_gradient_checkpointing
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        
        if self.use_checkpoint and self.training:
            return self._forward_with_checkpoint(x, mask, kv_cache, use_cache)
        else:
            return self._forward(x, mask, kv_cache, use_cache)
    
    def _forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        # Self-attention с residual connection
        attn_out, new_cache = self.attention(self.norm1(x), mask, kv_cache, use_cache)
        x = x + self.dropout(attn_out)
        
        # MoE с residual connection
        moe_out, lb_loss = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x, new_cache, lb_loss
    
    def _forward_with_checkpoint(self, x, mask, kv_cache, use_cache):
        """Forward с gradient checkpointing для экономии памяти"""
        from torch.utils.checkpoint import checkpoint
        return checkpoint(self._forward, x, mask, kv_cache, use_cache, use_reentrant=False)


class DecisionTransformer(nn.Module):
    """
    Улучшенный Decision Transformer с:
    - Return-to-go conditioning
    - MoE и GQA
    - RoPE
    - Flash Attention
    - KV-Cache
    - Load balancing
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Инициализация с поддержкой двух форматов:
        
        Вариант 1 (новый): DecisionTransformer(config=ModelConfig(...))
        Вариант 2 (старый): DecisionTransformer(obs_dim=100, action_dim=5, ...)
        """
        super().__init__()
        
        # Обратная совместимость: если config None, создаём из kwargs
        if config is None:
            # Старый формат: отдельные параметры
            config = ModelConfig(
                obs_dim=kwargs.get('obs_dim', 1000),
                action_dim=kwargs.get('action_dim', 5),
                embed_dim=kwargs.get('embed_dim', 256),
                num_layers=kwargs.get('num_layers', 6),
                num_heads=kwargs.get('num_heads', 8),
                num_kv_heads=kwargs.get('num_kv_heads', 4),
                num_experts=kwargs.get('num_experts', 8),
                expert_capacity=kwargs.get('expert_capacity', 1.25),
                top_k_experts=kwargs.get('top_k_experts', 2),
                max_seq_len=kwargs.get('max_seq_len', 512),
                dropout=kwargs.get('dropout', 0.1),
                use_flash_attn=kwargs.get('use_flash_attn', True),
                use_gradient_checkpointing=kwargs.get('use_gradient_checkpointing', False),
                rope_base=kwargs.get('rope_base', 10000),
                load_balancing_loss_coef=kwargs.get('load_balancing_loss_coef', 0.01)
            )
        
        self.config = config
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        
        # Encoders для RTG, observations и actions
        self.rtg_encoder = nn.Linear(1, config.embed_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(config.obs_dim, config.embed_dim),
            nn.Tanh()
        )
        self.action_embedding = nn.Embedding(config.action_dim, config.embed_dim)
        
        # Type embeddings: 0=RTG, 1=observation, 2=action
        self.type_embedding = nn.Embedding(3, config.embed_dim)
        
        # Трансформер блоки
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Выходная голова для предсказания действий
        self.action_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.action_dim)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self, 
        observations: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            observations: [batch, seq_len, obs_dim]
            actions: [batch, seq_len] - индексы действий
            returns_to_go: [batch, seq_len] - целевой return (опционально, для обратной совместимости)
            timesteps: [batch, seq_len] - временные метки (опционально)
            attention_mask: [batch, seq_len] - маска для padding
            
        Returns:
            action_logits: [batch, seq_len, action_dim]
            lb_loss: load balancing loss для MoE
        """
        batch_size, seq_len = observations.shape[0], observations.shape[1]
        
        # === ОБРАТНАЯ СОВМЕСТИМОСТЬ ===
        # Если returns_to_go не передан, используем placeholder (нули)
        if returns_to_go is None:
            returns_to_go = torch.zeros(batch_size, seq_len, device=observations.device)
        
        # Создаем чередующуюся последовательность: RTG, obs, action, RTG, obs, action, ...
        # Итоговая длина: 3 * seq_len
        max_len = 3 * seq_len
        
        # Embeddings для каждого типа токенов
        rtg_embeds = self.rtg_encoder(returns_to_go.unsqueeze(-1))  # [batch, seq_len, embed_dim]
        obs_embeds = self.obs_encoder(observations)  # [batch, seq_len, embed_dim]
        action_embeds = self.action_embedding(actions)  # [batch, seq_len, embed_dim]
        
        # Собираем последовательность
        token_embeddings = torch.zeros(
            batch_size, max_len, self.embed_dim,
            dtype=obs_embeds.dtype, device=obs_embeds.device
        )
        
        # RTG на позициях 0, 3, 6, ...
        token_embeddings[:, 0::3, :] = rtg_embeds
        # Observations на позициях 1, 4, 7, ...
        token_embeddings[:, 1::3, :] = obs_embeds
        # Actions на позициях 2, 5, 8, ...
        token_embeddings[:, 2::3, :] = action_embeds
        
        # Type embeddings
        type_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=obs_embeds.device)
        type_ids[:, 0::3] = 0  # RTG
        type_ids[:, 1::3] = 1  # observation
        type_ids[:, 2::3] = 2  # action
        type_embeds = self.type_embedding(type_ids)
        
        # Финальные embeddings
        x = token_embeddings + type_embeds
        x = self.dropout(x)
        
        # Causal mask
        causal_mask = torch.tril(
            torch.ones(max_len, max_len, device=x.device)
        ).unsqueeze(0).unsqueeze(0)
        
        # Проходим через трансформер блоки
        total_lb_loss = 0.0
        for block in self.blocks:
            x, _, lb_loss = block(x, mask=causal_mask, use_cache=False)
            total_lb_loss += lb_loss
        
        x = self.norm(x)
        
        # Предсказываем действия только на позициях observations (1, 4, 7, ...)
        # Берем embeddings observations и предсказываем следующие действия
        obs_positions = torch.arange(1, max_len, 3, device=x.device)
        x_obs = x[:, obs_positions, :]  # [batch, seq_len, embed_dim]
        
        action_logits = self.action_head(x_obs)
        
        # Средний load balancing loss по всем слоям
        avg_lb_loss = total_lb_loss / len(self.blocks)
        
        return action_logits, avg_lb_loss
    
    @torch.no_grad()
    def get_action(
        self, 
        observations: torch.Tensor,
        returns_to_go: torch.Tensor,
        past_actions: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Предсказывает действие для текущего observation
        
        Args:
            observations: [batch, history_len, obs_dim] или [batch, 1, obs_dim]
            returns_to_go: [batch, history_len] или [batch, 1]
            past_actions: [batch, history_len-1] или None
            temperature: температура для sampling
            top_k: top-k sampling
            top_p: nucleus sampling
            
        Returns:
            action: [batch] - предсказанные действия
        """
        batch_size = observations.shape[0]
        
        if past_actions is None:
            # Холодный старт: инициализируем случайными действиями
            past_actions = torch.zeros(
                batch_size, 0, 
                dtype=torch.long, device=observations.device
            )
        
        # Если у нас только последний observation, нужна вся история
        if observations.shape[1] == 1 and past_actions.shape[1] > 0:
            raise ValueError("Нужна полная история observations для контекста")
        
        # Дополняем actions до нужной длины (может быть меньше на 1)
        seq_len = observations.shape[1]
        if past_actions.shape[1] < seq_len:
            # Дополняем нулями (placeholder)
            padding = torch.zeros(
                batch_size, seq_len - past_actions.shape[1],
                dtype=torch.long, device=observations.device
            )
            actions = torch.cat([past_actions, padding], dim=1)
        else:
            actions = past_actions[:, :seq_len]
        
        # Forward pass
        action_logits, _ = self.forward(observations, actions, returns_to_go)
        
        # Берем логиты для последней позиции
        last_logits = action_logits[:, -1, :] / temperature
        
        # Top-k sampling
        if top_k is not None:
            indices_to_remove = last_logits < torch.topk(last_logits, top_k)[0][..., -1, None]
            last_logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) sampling
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            last_logits[indices_to_remove] = float('-inf')
        
        # Sampling
        probs = F.softmax(last_logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return action


def create_model(config=None, **kwargs) -> DecisionTransformer:
    """
    Фабрика для создания модели с поддержкой двух форматов
    
    Вариант 1: create_model(config=ModelConfig(...))
    Вариант 2: create_model(obs_dim=100, action_dim=5, ...)
    """
    if config is None and len(kwargs) > 0:
        # Создаём config из kwargs
        config = ModelConfig(**kwargs)
    elif config is None:
        # Дефолтная конфигурация
        config = ModelConfig()
    
    model = DecisionTransformer(config=config)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Создана модель Decision Transformer:")
    print(f"  Всего параметров: {total_params:,}")
    print(f"  Обучаемых параметров: {trainable_params:,}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads} (KV heads: {config.num_kv_heads})")
    print(f"  Experts: {config.num_experts} (top-{config.top_k_experts})")
    print(f"  Flash Attention: {config.use_flash_attn and HAS_FLASH_ATTN}")
    print(f"  SDPA: {HAS_SDPA}")
    
    return model

# Пример использования
if __name__ == "__main__":
    # Конфигурация модели
    config = ModelConfig(
        obs_dim=100,
        action_dim=5,
        embed_dim=128,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        num_experts=4,
        top_k_experts=2,
        max_seq_len=256,
        dropout=0.1,
        use_flash_attn=True,
        use_gradient_checkpointing=False
    )
    
    # Создаем модель
    model = create_model(config)
    model = model.cuda() if torch.cuda.is_available() else model
    
    # Тестовые данные
    batch_size = 4
    seq_len = 10
    
    observations = torch.randn(batch_size, seq_len, config.obs_dim)
    actions = torch.randint(0, config.action_dim, (batch_size, seq_len))
    returns_to_go = torch.randn(batch_size, seq_len)
    
    if torch.cuda.is_available():
        observations = observations.cuda()
        actions = actions.cuda()
        returns_to_go = returns_to_go.cuda()
    
    # Forward pass
    action_logits, lb_loss = model(observations, actions, returns_to_go)
    
    print(f"\nВыход модели:")
    print(f"  Action logits shape: {action_logits.shape}")
    print(f"  Load balancing loss: {lb_loss.item():.6f}")
    
    # Инференс
    test_obs = observations[:, -1:, :]
    test_rtg = returns_to_go[:, -1:]
    predicted_action = model.get_action(
        test_obs, 
        test_rtg,
        temperature=1.0,
        top_k=3
    )
    print(f"  Predicted actions: {predicted_action}")