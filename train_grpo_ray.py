"""
GRPO (Group Relative Policy Optimization) Trainer with Ray Parallelization
Второй этап обучения Decision Transformer с подкреплением

УЛУЧШЕНИЯ:
1. Ray для параллельного сбора роллаутов (6-8x ускорение)
2. Асинхронное выполнение сред на разных CPU cores
3. Эффективная передача данных через shared memory
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
from tqdm import tqdm
from collections import deque

# Ray для параллелизации
import ray
from ray.util.queue import Queue

from model import DecisionTransformer


class ValueNetwork(nn.Module):
    """
    Critic network для оценки value функции
    """
    
    def __init__(self, obs_dim, embed_dim=256, hidden_dim=512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs):
        """
        Args:
            obs: [batch, seq_len, obs_dim] или [batch, obs_dim]
        Returns:
            values: [batch, seq_len, 1] или [batch, 1]
        """
        return self.network(obs)


class RolloutBuffer:
    """
    Буфер для хранения траекторий (rollouts)
    """
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get(self):
        # Оптимизация: сначала конвертируем в numpy arrays, затем в тензоры
        return (
            torch.from_numpy(np.array(self.observations)).float(),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.dones)
        )
    
    def __len__(self):
        return len(self.observations)


@ray.remote
class RolloutWorker:
    """
    Ray worker для параллельного сбора роллаутов
    Каждый worker управляет своей средой и собирает траектории
    """
    
    def __init__(self, env_fn, policy_state_dict, value_state_dict, config, worker_id):
        """
        Args:
            env_fn: функция создания среды
            policy_state_dict: веса policy для загрузки
            value_state_dict: веса value для загрузки
            config: конфигурация
            worker_id: ID worker'а
        """
        self.worker_id = worker_id
        self.config = config
        self.device = 'cpu'  # Workers работают на CPU
        
        # Создаем среду
        self.env = env_fn()
        
        # Загружаем модели
        self.policy = DecisionTransformer(
            obs_dim=config['obs_dim'],
            action_dim=config['action_dim'],
            embed_dim=config['embed_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            num_kv_heads=config['num_kv_heads'],
            num_experts=config['num_experts'],
            max_seq_len=config.get('max_seq_len', 512),
            dropout=config.get('dropout', 0.1)
        ).to(self.device)
        self.policy.load_state_dict(policy_state_dict)
        self.policy.eval()
        
        self.value_net = ValueNetwork(
            obs_dim=config['obs_dim'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['embed_dim'] * 2
        ).to(self.device)
        self.value_net.load_state_dict(value_state_dict)
        self.value_net.eval()
        
        # История для контекста
        self.context_length = config['context_length']
        self.obs_history = deque(maxlen=self.context_length)
        self.action_history = deque(maxlen=self.context_length - 1)
        
        # Текущее состояние
        self.current_obs = self.env.reset()
        self.obs_history.append(self.current_obs)
        
        self.episode_reward = 0.0
        self.episode_length = 0
        
    def collect_steps(self, num_steps):
        """
        Собирает num_steps шагов в среде
        
        Returns:
            buffer: RolloutBuffer с собранными данными
            stats: статистика (завершенные эпизоды)
        """
        buffer = RolloutBuffer()
        completed_episodes = []
        
        with torch.no_grad():
            for step in range(num_steps):
                # Подготовка последовательности для policy
                obs_array = np.array(list(self.obs_history))
                obs_seq = torch.from_numpy(obs_array).float().unsqueeze(0)  # [1, hist_len, obs_dim]
                
                if len(self.action_history) > 0:
                    actions_array = np.array(list(self.action_history))
                    actions_seq = torch.from_numpy(actions_array).long()
                    start_token = torch.LongTensor([self.policy.action_dim])
                    actions_seq = torch.cat([start_token, actions_seq]).unsqueeze(0)
                else:
                    actions_seq = torch.full((1, len(self.obs_history)), 
                                            self.policy.action_dim, dtype=torch.long)
                
                # Policy forward
                action_logits = self.policy(obs_seq, actions_seq)
                last_logits = action_logits[0, -1, :]
                
                # Sample action
                action_probs = torch.softmax(last_logits, dim=-1)
                action = torch.multinomial(action_probs, num_samples=1).item()
                log_prob = torch.log(action_probs[action]).item()
                
                # Value estimation
                current_obs_tensor = torch.from_numpy(self.current_obs).float().unsqueeze(0)
                value = self.value_net(current_obs_tensor).squeeze().item()
                
                # Step в среде
                next_obs, reward, done, info = self.env.step(action)
                
                # Сохраняем в буфер
                buffer.add(
                    self.current_obs,
                    action,
                    reward,
                    value,
                    log_prob,
                    float(done)
                )
                
                self.episode_reward += reward
                self.episode_length += 1
                
                if done:
                    # Эпизод завершен
                    completed_episodes.append({
                        'reward': self.episode_reward,
                        'length': self.episode_length,
                        'worker_id': self.worker_id
                    })
                    
                    # Reset
                    next_obs = self.env.reset()
                    self.obs_history.clear()
                    self.action_history.clear()
                    self.episode_reward = 0.0
                    self.episode_length = 0
                    # После reset добавляем только наблюдение
                    self.obs_history.append(next_obs)
                else:
                    # Обновляем историю только если эпизод продолжается
                    self.obs_history.append(next_obs)
                    self.action_history.append(action)
                
                self.current_obs = next_obs
        
        return buffer, completed_episodes
    
    def update_models(self, policy_state_dict, value_state_dict):
        """Обновляет веса моделей"""
        self.policy.load_state_dict(policy_state_dict)
        self.value_net.load_state_dict(value_state_dict)
    
    def close(self):
        """Закрывает среду"""
        self.env.close()


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) Trainer with Ray parallelization
    """
    
    def __init__(
        self,
        policy_model,  # DecisionTransformer
        value_model,   # ValueNetwork
        env_fn,        # Функция создания среды
        config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_ray=True   # Использовать ли Ray для параллелизации
    ):
        self.device = device
        self.config = config
        self.use_ray = use_ray
        
        # Проверка Windows для torch.compile
        is_windows = sys.platform.startswith('win')
        if is_windows and config.get('use_compile', False):
            print("⚠️  Windows: torch.compile отключен")
            config['use_compile'] = False
        
        # Модели
        if config.get('use_compile', False) and hasattr(torch, 'compile'):
            print("🚀 Компилируем policy с torch.compile...")
            policy_model = torch.compile(policy_model, mode='reduce-overhead')
        
        self.policy = policy_model.to(device)
        self.value_net = value_model.to(device)
        
        # Оптимизаторы
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config['policy_lr'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.value_optimizer = optim.AdamW(
            self.value_net.parameters(),
            lr=config['value_lr'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Среда
        self.env_fn = env_fn
        
        # Параметры GRPO
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Параметры группировки для GRPO
        self.group_size = config.get('group_size', 8)
        
        # Rollout параметры
        self.rollout_steps = config.get('rollout_steps', 128)
        self.num_workers = config.get('num_envs', 8)
        self.context_length = config.get('context_length', 20)
        
        # Обучение
        self.num_epochs = config.get('num_epochs', 4)
        self.batch_size = config.get('batch_size', 64)
        
        # Mixed Precision
        self.use_amp = config.get('use_amp', False) and device == 'cuda'
        if self.use_amp:
            # Используем старый API, т.к. новый API имеет проблемы совместимости
            self.scaler = GradScaler()
        
        # Логирование
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        self.global_step = 0
        
        # Статистика
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Создаем директории
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # Сохраняем конфиг
        with open(os.path.join(config['checkpoint_dir'], 'grpo_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Ray инициализация
        if self.use_ray:
            self._init_ray_workers()
        
        print(f"\n{'='*80}")
        print("GRPO Trainer инициализирован")
        print(f"{'='*80}")
        print(f"  Device: {self.device}")
        print(f"  Num workers: {self.num_workers}")
        print(f"  Use Ray: {self.use_ray}")
        print(f"  Rollout steps: {self.rollout_steps}")
        print(f"  Context length: {self.context_length}")
        print(f"  Group size: {self.group_size}")
        print(f"  Clip epsilon: {self.clip_epsilon}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"{'='*80}\n")
    
    def _init_ray_workers(self):
        """Инициализация Ray workers"""
        if not ray.is_initialized():
            # Инициализируем Ray с ограничением на CPU
            ray.init(
                num_cpus=self.num_workers + 2,  # +2 для главного процесса
                ignore_reinit_error=True,
                log_to_driver=False
            )
            print(f"✓ Ray инициализирован")
        
        # Получаем веса моделей для передачи workers
        # ВАЖНО: Workers работают на CPU, поэтому перемещаем веса на CPU
        policy_to_share = self.policy._orig_mod if hasattr(self.policy, '_orig_mod') else self.policy
        policy_state_dict = {k: v.cpu() for k, v in policy_to_share.state_dict().items()}
        value_state_dict = {k: v.cpu() for k, v in self.value_net.state_dict().items()}
        
        # Конфигурация для workers
        worker_config = {
            'obs_dim': self.config['obs_dim'],
            'action_dim': self.config['action_dim'],
            'embed_dim': self.config['embed_dim'],
            'num_layers': self.config['num_layers'],
            'num_heads': self.config['num_heads'],
            'num_kv_heads': self.config['num_kv_heads'],
            'num_experts': self.config['num_experts'],
            'max_seq_len': self.config.get('max_seq_len', 512),
            'dropout': self.config.get('dropout', 0.1),
            'context_length': self.context_length
        }
        
        # Создаем workers
        self.workers = [
            RolloutWorker.remote(
                self.env_fn,
                policy_state_dict,
                value_state_dict,
                worker_config,
                worker_id=i
            )
            for i in range(self.num_workers)
        ]
        
        print(f"✓ Создано {self.num_workers} Ray workers")
    
    def _update_workers(self):
        """Обновляет веса моделей во всех workers"""
        # ВАЖНО: Workers работают на CPU, поэтому перемещаем веса на CPU
        policy_to_share = self.policy._orig_mod if hasattr(self.policy, '_orig_mod') else self.policy
        policy_state_dict = {k: v.cpu() for k, v in policy_to_share.state_dict().items()}
        value_state_dict = {k: v.cpu() for k, v in self.value_net.state_dict().items()}
        
        # Асинхронно обновляем всех workers
        ray.get([
            worker.update_models.remote(policy_state_dict, value_state_dict)
            for worker in self.workers
        ])
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: [num_steps]
            values: [num_steps]
            dones: [num_steps]
            next_value: scalar
            
        Returns:
            advantages: [num_steps]
            returns: [num_steps]
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def compute_group_advantages(self, advantages):
        """
        Compute group relative advantages (ключевая часть GRPO)
        
        Группируем advantages и нормализуем относительно группы
        """
        # Reshape в группы
        num_samples = len(advantages)
        num_groups = num_samples // self.group_size
        
        if num_groups == 0:
            # Если меньше одной группы, используем обычную нормализацию
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Обрезаем до кратного group_size
        advantages = advantages[:num_groups * self.group_size]
        advantages = advantages.view(num_groups, self.group_size)
        
        # Нормализуем внутри каждой группы
        group_mean = advantages.mean(dim=1, keepdim=True)
        group_std = advantages.std(dim=1, keepdim=True)
        
        normalized = (advantages - group_mean) / (group_std + 1e-8)
        
        return normalized.flatten()
    
    def collect_rollouts_ray(self):
        """
        Собирает rollouts параллельно используя Ray workers
        """
        # Запускаем сбор роллаутов на всех workers параллельно
        steps_per_worker = self.rollout_steps // self.num_workers
        
        futures = [
            worker.collect_steps.remote(steps_per_worker)
            for worker in self.workers
        ]
        
        # Ждем завершения всех workers
        results = ray.get(futures)
        
        # Объединяем результаты
        buffers = []
        for buffer, completed_episodes in results:
            buffers.append(buffer)
            # Добавляем статистику завершенных эпизодов
            for ep in completed_episodes:
                self.episode_rewards.append(ep['reward'])
                self.episode_lengths.append(ep['length'])
        
        return buffers
    
    def collect_rollouts_sequential(self):
        """
        Собирает rollouts последовательно (fallback если Ray не используется)
        """
        # Создаем среды
        envs = [self.env_fn() for _ in range(self.num_workers)]
        
        # Буферы для каждой среды
        buffers = [RolloutBuffer() for _ in range(self.num_workers)]
        
        # Инициализация
        observations = [env.reset() for env in envs]
        episode_rewards = [0.0] * self.num_workers
        episode_lengths = [0] * self.num_workers
        
        # История для контекста
        obs_history = [deque([obs], maxlen=self.context_length) for obs in observations]
        action_history = [deque(maxlen=self.context_length-1) for _ in range(self.num_workers)]
        
        self.policy.eval()
        self.value_net.eval()
        
        with torch.no_grad():
            for step in range(self.rollout_steps):
                # Подготовка batch
                batch_obs_seq = []
                batch_actions_seq = []
                
                for i in range(self.num_workers):
                    # Конвертируем deque в numpy array сначала для оптимизации
                    obs_array = np.array(list(obs_history[i]))  # [hist_len, obs_dim]
                    obs_seq = torch.from_numpy(obs_array).float().unsqueeze(0)  # [1, hist_len, obs_dim]
                    
                    if len(action_history[i]) > 0:
                        actions_array = np.array(list(action_history[i]))
                        actions_seq = torch.from_numpy(actions_array).long()
                        # Добавляем START token в начало
                        start_token = torch.LongTensor([self.policy.action_dim])
                        actions_seq = torch.cat([start_token, actions_seq]).unsqueeze(0)  # [1, hist_len]
                    else:
                        # Только START token
                        actions_seq = torch.full((1, len(obs_history[i])), self.policy.action_dim, dtype=torch.long)
                    
                    batch_obs_seq.append(obs_seq)
                    batch_actions_seq.append(actions_seq)
                
                # Pad до одной длины
                max_len = max(obs.size(1) for obs in batch_obs_seq)
                
                padded_obs = []
                padded_actions = []
                
                for obs_seq, actions_seq in zip(batch_obs_seq, batch_actions_seq):
                    if obs_seq.size(1) < max_len:
                        pad_len = max_len - obs_seq.size(1)
                        obs_pad = torch.zeros(1, pad_len, obs_seq.size(2))
                        obs_seq = torch.cat([obs_pad, obs_seq], dim=1)
                        
                        action_pad = torch.full((1, pad_len), self.policy.action_dim, dtype=torch.long)
                        actions_seq = torch.cat([action_pad, actions_seq], dim=1)
                    
                    padded_obs.append(obs_seq)
                    padded_actions.append(actions_seq)
                
                batch_obs = torch.cat(padded_obs, dim=0).to(self.device)  # [num_envs, max_len, obs_dim]
                batch_actions = torch.cat(padded_actions, dim=0).to(self.device)  # [num_envs, max_len]
                
                # Policy forward
                action_logits = self.policy(batch_obs, batch_actions)
                last_logits = action_logits[:, -1, :]  # [num_envs, action_dim]
                
                # Sample actions
                action_probs = torch.softmax(last_logits, dim=-1)
                actions = torch.multinomial(action_probs, num_samples=1).squeeze(-1)  # [num_envs]
                log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))  # [num_envs]
                
                # Value estimation (используем текущее наблюдение)
                # Конвертируем список numpy массивов в один массив для оптимизации
                current_obs_array = np.array(observations)  # [num_envs, obs_dim]
                current_obs = torch.from_numpy(current_obs_array).float().to(self.device)
                values = self.value_net(current_obs).squeeze(-1)  # [num_envs]
                
                # Step в средах
                actions_np = actions.cpu().numpy()
                
                for i in range(self.num_workers):
                    next_obs, reward, done, info = envs[i].step(actions_np[i])
                    
                    # Сохраняем в буфер
                    buffers[i].add(
                        observations[i],
                        actions_np[i],
                        reward,
                        values[i].item(),
                        log_probs[i].item(),
                        float(done)
                    )
                    
                    episode_rewards[i] += reward
                    episode_lengths[i] += 1
                    
                    if done:
                        # Эпизод закончен
                        self.episode_rewards.append(episode_rewards[i])
                        self.episode_lengths.append(episode_lengths[i])
                        
                        # Reset
                        next_obs = envs[i].reset()
                        obs_history[i].clear()
                        action_history[i].clear()
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0
                        # После reset добавляем только наблюдение, без действия
                        obs_history[i].append(next_obs)
                    else:
                        # Обновляем историю только если эпизод продолжается
                        obs_history[i].append(next_obs)
                        action_history[i].append(actions_np[i])
                    
                    observations[i] = next_obs
        
        # Закрываем среды
        for env in envs:
            env.close()
        
        return buffers
    
    def collect_rollouts(self):
        """
        Собирает rollouts (выбирает метод в зависимости от use_ray)
        """
        if self.use_ray:
            return self.collect_rollouts_ray()
        else:
            return self.collect_rollouts_sequential()
    
    def update_policy(self, buffers):
        """
        Обновляет policy и value network используя GRPO
        """
        # Собираем все данные из буферов
        all_obs = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_old_log_probs = []
        
        for buffer in buffers:
            if len(buffer) == 0:
                continue
            
            obs, actions, rewards, values, log_probs, dones = buffer.get()
            
            # Compute GAE
            with torch.no_grad():
                # obs уже является тензором из buffer.get()
                last_obs = obs[-1].unsqueeze(0).to(self.device)  # [1, obs_dim]
                next_value = self.value_net(last_obs).squeeze().item()
            
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            
            all_obs.append(obs)
            all_actions.append(actions)
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_old_log_probs.append(log_probs)
        
        # Concatenate
        all_obs = torch.cat(all_obs, dim=0).to(self.device)
        all_actions = torch.cat(all_actions, dim=0).to(self.device)
        all_advantages = torch.cat(all_advantages, dim=0).to(self.device)
        all_returns = torch.cat(all_returns, dim=0).to(self.device)
        all_old_log_probs = torch.cat(all_old_log_probs, dim=0).to(self.device)
        
        # Group relative advantages (GRPO!)
        all_advantages = self.compute_group_advantages(all_advantages)
        
        # Множественные эпохи обучения на собранных данных
        dataset_size = len(all_obs)
        indices = np.arange(dataset_size)
        
        policy_losses = []
        value_losses = []
        entropies = []
        
        self.policy.train()
        self.value_net.train()
        
        for epoch in range(self.num_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = all_obs[batch_indices]
                batch_actions = all_actions[batch_indices]
                batch_advantages = all_advantages[batch_indices]
                batch_returns = all_returns[batch_indices]
                batch_old_log_probs = all_old_log_probs[batch_indices]
                
                # Для policy нужна последовательность, создаем простую (без истории для упрощения)
                obs_seq = batch_obs.unsqueeze(1)  # [batch, 1, obs_dim]
                actions_seq = torch.full((len(batch_obs), 1), self.policy.action_dim, 
                                        dtype=torch.long, device=self.device)  # START tokens
                
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        # Policy forward
                        action_logits = self.policy(obs_seq, actions_seq)
                        action_probs = torch.softmax(action_logits[:, -1, :], dim=-1)
                        
                        dist = torch.distributions.Categorical(action_probs)
                        new_log_probs = dist.log_prob(batch_actions)
                        entropy = dist.entropy()
                        
                        # Policy loss (GRPO with clipping)
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Entropy bonus
                        entropy_loss = -self.entropy_coef * entropy.mean()
                        
                        # Value loss
                        values = self.value_net(batch_obs).squeeze()
                        value_loss = self.value_coef * ((values - batch_returns) ** 2).mean()
                        
                        # Total loss
                        total_loss = policy_loss + value_loss + entropy_loss
                    
                    # Backward
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    
                    self.scaler.scale(total_loss).backward()
                    
                    self.scaler.unscale_(self.policy_optimizer)
                    self.scaler.unscale_(self.value_optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(self.policy_optimizer)
                    self.scaler.step(self.value_optimizer)
                    self.scaler.update()
                    
                else:
                    # Policy forward
                    action_logits = self.policy(obs_seq, actions_seq)
                    action_probs = torch.softmax(action_logits[:, -1, :], dim=-1)
                    
                    dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()
                    
                    # Policy loss (GRPO with clipping)
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Entropy bonus
                    entropy_loss = -self.entropy_coef * entropy.mean()
                    
                    # Value loss
                    values = self.value_net(batch_obs).squeeze()
                    value_loss = self.value_coef * ((values - batch_returns) ** 2).mean()
                    
                    # Total loss
                    total_loss = policy_loss + value_loss + entropy_loss
                    
                    # Backward
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    total_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                    
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies)
        }
    
    def train(self, num_iterations):
        """
        Основной цикл обучения GRPO
        """
        print("="*80)
        print("НАЧИНАЕМ GRPO ОБУЧЕНИЕ" + (" (с Ray)" if self.use_ray else ""))
        print("="*80)
        print(f"Iterations: {num_iterations}")
        print(f"Steps per iteration: {self.rollout_steps}")
        print(f"Workers: {self.num_workers}")
        print("="*80)
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            iter_start = time.time()
            
            # Обновляем веса в workers (только для Ray)
            if self.use_ray:
                self._update_workers()
            
            # Collect rollouts
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            print("Collecting rollouts...")
            buffers = self.collect_rollouts()
            
            # Update policy
            print("Updating policy...")
            metrics = self.update_policy(buffers)
            
            iter_time = time.time() - iter_start
            
            # Логирование
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards)
                mean_length = np.mean(self.episode_lengths)
            else:
                mean_reward = 0.0
                mean_length = 0.0
            
            self.writer.add_scalar('train/policy_loss', metrics['policy_loss'], iteration)
            self.writer.add_scalar('train/value_loss', metrics['value_loss'], iteration)
            self.writer.add_scalar('train/entropy', metrics['entropy'], iteration)
            self.writer.add_scalar('train/mean_reward', mean_reward, iteration)
            self.writer.add_scalar('train/mean_length', mean_length, iteration)
            self.writer.add_scalar('train/iteration_time', iter_time, iteration)
            
            print(f"\nIteration {iteration + 1} Results:")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Mean Reward: {mean_reward:.2f}")
            print(f"  Mean Length: {mean_length:.1f}")
            print(f"  Time: {iter_time:.2f}s")
            print("-"*80)
            
            # Сохранение чекпоинтов
            if (iteration + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'iteration_{iteration + 1}.pt')
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("GRPO ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"{'='*80}")
        print(f"Общее время: {total_time / 3600:.2f} часов")
        print(f"Среднее время на итерацию: {total_time / num_iterations:.2f} секунд")
        print(f"{'='*80}")
        
        self.writer.close()
        
        # Закрываем Ray workers
        if self.use_ray:
            ray.get([worker.close.remote() for worker in self.workers])
    
    def save_checkpoint(self, filename):
        """Сохранение чекпоинта"""
        # Если модель скомпилирована, берем оригинал
        policy_to_save = self.policy._orig_mod if hasattr(self.policy, '_orig_mod') else self.policy
        
        checkpoint = {
            'policy_state_dict': policy_to_save.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config
        }
        
        path = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save(checkpoint, path)
        print(f"✓ Чекпоинт сохранен: {path}")
    
    def __del__(self):
        """Очистка Ray при удалении тренера"""
        if self.use_ray and ray.is_initialized():
            # Закрываем workers
            if hasattr(self, 'workers'):
                try:
                    ray.get([worker.close.remote() for worker in self.workers])
                except:
                    pass