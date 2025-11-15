import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - компромисс между MHA и MQA
    """
    
    def __init__(self, embed_dim, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim должен делиться на num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        # Q проекции для всех heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        
        # K, V проекции для меньшего количества KV heads
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Q: [batch, seq_len, num_heads, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # K, V: [batch, seq_len, num_kv_heads, head_dim]
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Повторяем KV для каждой группы Q heads
        # [batch, seq_len, num_kv_heads, head_dim] -> [batch, seq_len, num_heads, head_dim]
        k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=2)
        
        # Transpose для attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # [batch, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        output = self.out_proj(attn_output)
        return output


class MoELayer(nn.Module):
    """
    Mixture of Experts layer
    """
    
    def __init__(self, embed_dim, num_experts=8, expert_dim=None, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        if expert_dim is None:
            expert_dim = 4 * embed_dim
        
        # Router: определяет какие эксперты использовать
        self.router = nn.Linear(embed_dim, num_experts)
        
        # Эксперты - простые FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, embed_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Flatten для router
        x_flat = x.view(-1, embed_dim)  # [batch*seq_len, embed_dim]
        
        # Router scores
        router_logits = self.router(x_flat)  # [batch*seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Выбираем top_k экспертов
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Нормализуем веса top_k экспертов
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Проходим через экспертов и взвешиваем их выходы
        output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_weights = top_k_probs[:, i].unsqueeze(-1)
            
            # Для каждого эксперта применяем только к соответствующим токенам
            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_weights[mask] * expert_output
        
        # Reshape обратно
        output = output.view(batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(nn.Module):
    """
    Transformer block с GQA и MoE
    """
    
    def __init__(self, embed_dim, num_heads, num_kv_heads, num_experts=8, dropout=0.1):
        super().__init__()
        
        self.attention = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, dropout)
        self.moe = MoELayer(embed_dim, num_experts, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention с residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # MoE с residual connection
        moe_out = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x


class DecisionTransformer(nn.Module):
    """
    Decision Transformer с MoE и GQA для RL
    
    Архитектура:
    - Входная последовательность: [START, obs_0, act_0, obs_1, act_1, ...]
    - Модель предсказывает следующее действие
    """
    
    def __init__(
        self,
        obs_dim=1000,
        action_dim=5,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        num_kv_heads=4,
        num_experts=8,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Проекции для входов
        self.obs_encoder = nn.Linear(obs_dim, embed_dim)
        self.action_embedding = nn.Embedding(action_dim + 1, embed_dim)  # +1 для START токена
        
        # Позиционные эмбеддинги
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Type embeddings (для различения obs и action)
        self.type_embedding = nn.Embedding(3, embed_dim)  # 0: START, 1: obs, 2: action
        
        # Трансформер блоки
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, num_kv_heads, num_experts, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Выходная голова для предсказания действий
        self.action_head = nn.Linear(embed_dim, action_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Инициализация
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, observations, actions, attention_mask=None):
        """
        Args:
            observations: [batch, seq_len, obs_dim]
            actions: [batch, seq_len] (может содержать START токен = action_dim)
            attention_mask: [batch, seq_len] опционально
            
        Returns:
            action_logits: [batch, seq_len, action_dim]
        """
        batch_size, seq_len = actions.shape
        
        # Encode observations
        obs_embeds = self.obs_encoder(observations)  # [batch, seq_len, embed_dim]
        
        # Embed actions
        action_embeds = self.action_embedding(actions)  # [batch, seq_len, embed_dim]
        
        # Чередуем obs и actions в последовательности
        # Позиции: [0: START/obs_0, 1: act_0, 2: obs_1, 3: act_1, ...]
        # Простой подход: складываем obs и action embeddings
        token_embeds = obs_embeds + action_embeds
        
        # Type embeddings (0: START, 1: obs, 2: action)
        # Для простоты используем pattern: четные позиции - obs, нечетные - actions
        type_ids = torch.ones(batch_size, seq_len, dtype=torch.long, device=actions.device)
        type_ids[:, 0] = 0  # START token
        type_embeds = self.type_embedding(type_ids)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=actions.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)
        
        # Комбинируем все embeddings
        x = token_embeds + type_embeds + pos_embeds
        x = self.dropout(x)
        
        # Causal mask (автореgressивная маска)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        # Проходим через трансформер блоки
        for block in self.blocks:
            x = block(x, mask=causal_mask)
        
        x = self.norm(x)
        
        # Предсказываем действия
        action_logits = self.action_head(x)
        
        return action_logits
    
    def get_action(self, observations, past_actions=None, temperature=1.0):
        """
        Получить действие для текущего observation с учетом истории
        
        Args:
            observations: [batch, history_len, obs_dim]
            past_actions: [batch, history_len-1] или None для холодного старта
            temperature: float для sampling
            
        Returns:
            action: [batch] - predicted action indices
        """
        batch_size, history_len, _ = observations.shape
        
        if past_actions is None:
            # Холодный старт: используем START токен
            actions = torch.full((batch_size, history_len), self.action_dim, 
                               dtype=torch.long, device=observations.device)
        else:
            # Добавляем START токен в начало
            start_token = torch.full((batch_size, 1), self.action_dim, 
                                    dtype=torch.long, device=observations.device)
            actions = torch.cat([start_token, past_actions], dim=1)
        
        # Forward pass
        with torch.no_grad():
            action_logits = self.forward(observations, actions)
        
        # Берем логиты для последней позиции
        last_logits = action_logits[:, -1, :] / temperature
        
        # Sampling
        probs = F.softmax(last_logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return action
