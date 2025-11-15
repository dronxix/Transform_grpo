"""
Подготовка данных из Gymnasium сред для обучения Decision Transformer
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from gym_environment import (
    create_gym_environment, 
    GymnasiumTrajectoryCollector,
    PRESET_ENVS
)


class RLDataset(Dataset):
    """
    Dataset для обучения Decision Transformer с Return-To-Go conditioning
    
    Создает последовательности (rtg_0, obs_0, act_0, rtg_1, obs_1, act_1, ...) из траекторий
    """
    
    def __init__(self, trajectories, context_length=20, action_dim=5, gamma=1.0):
        """
        Args:
            trajectories: list of dicts с ключами 'observations', 'actions', 'rewards'
            context_length: длина контекста (сколько шагов истории использовать)
            action_dim: размерность пространства действий
            gamma: discount factor для вычисления returns (1.0 = недисконтированный)
        """
        self.trajectories = trajectories
        self.context_length = context_length
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Предварительно вычисляем returns-to-go для всех траекторий
        self._compute_returns_to_go()
        
        # Создаем индекс для быстрого доступа
        self._build_index()
        
    def _compute_returns_to_go(self):
        """
        Вычисляет returns-to-go для каждой траектории
        
        Returns-to-go (RTG) в timestep t = сумма всех будущих наград с t до конца эпизода
        RTG_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        """
        print("Вычисляем returns-to-go для траекторий...")
        
        for traj in self.trajectories:
            rewards = np.array(traj['rewards'], dtype=np.float32)
            
            # Вычисляем returns-to-go (идем с конца траектории)
            returns_to_go = np.zeros_like(rewards)
            running_return = 0.0
            
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + self.gamma * running_return
                returns_to_go[t] = running_return
            
            # Сохраняем в траекторию
            traj['returns_to_go'] = returns_to_go
        
        print(f"✓ Returns-to-go вычислены для {len(self.trajectories)} траекторий")
        
        # Статистика для нормализации (опционально)
        all_rtg = np.concatenate([traj['returns_to_go'] for traj in self.trajectories])
        self.rtg_mean = np.mean(all_rtg)
        self.rtg_std = np.std(all_rtg) + 1e-6
        
        print(f"  RTG статистика: mean={self.rtg_mean:.2f}, std={self.rtg_std:.2f}")
        print(f"  RTG диапазон: [{all_rtg.min():.2f}, {all_rtg.max():.2f}]")
        
    def _build_index(self):
        """Строим индекс всех возможных сэмплов"""
        self.samples = []
        
        for traj_idx, traj in enumerate(self.trajectories):
            traj_len = len(traj['actions'])
            
            # Для каждой позиции в траектории создаем возможный сэмпл
            for end_idx in range(1, traj_len + 1):
                self.samples.append({
                    'traj_idx': traj_idx,
                    'end_idx': end_idx
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Возвращает последовательность наблюдений, действий и returns-to-go
        
        Returns:
            observations: [seq_len, obs_dim]
            actions: [seq_len] - действия (с START токеном)
            returns_to_go: [seq_len] - целевые returns для conditioning
            targets: [seq_len] - целевые действия для обучения
            mask: [seq_len] - маска валидных позиций
        """
        sample = self.samples[idx]
        traj_idx = sample['traj_idx']
        end_idx = sample['end_idx']
        
        traj = self.trajectories[traj_idx]
        
        # Определяем начальную позицию (берем context_length элементов или меньше)
        start_idx = max(0, end_idx - self.context_length)
        
        # Извлекаем подпоследовательность
        obs_seq = traj['observations'][start_idx:end_idx]
        act_seq = traj['actions'][start_idx:end_idx]
        rtg_seq = traj['returns_to_go'][start_idx:end_idx]
        
        # Конвертируем в numpy
        observations = np.array(obs_seq, dtype=np.float32)
        actions = np.array(act_seq, dtype=np.int64)
        returns_to_go = np.array(rtg_seq, dtype=np.float32)
        
        seq_len = len(actions)
        
        # Паддинг если последовательность короче context_length
        if seq_len < self.context_length:
            pad_len = self.context_length - seq_len
            
            # Паддинг для observations (используем нули)
            obs_pad = np.zeros((pad_len, observations.shape[1]), dtype=np.float32)
            observations = np.concatenate([obs_pad, observations], axis=0)
            
            # Паддинг для actions (используем action_dim как START токен)
            act_pad = np.full((pad_len,), self.action_dim, dtype=np.int64)
            actions = np.concatenate([act_pad, actions], axis=0)
            
            # Паддинг для returns-to-go (используем нули или можно использовать среднее)
            rtg_pad = np.zeros((pad_len,), dtype=np.float32)
            returns_to_go = np.concatenate([rtg_pad, returns_to_go], axis=0)
            
            # Маска: 0 для паддинга, 1 для реальных данных
            mask = np.concatenate([np.zeros(pad_len), np.ones(seq_len)], axis=0)
        else:
            mask = np.ones(seq_len)
        
        # ВАЖНО: Убедимся что длины совпадают
        assert len(observations) == self.context_length
        assert len(actions) == self.context_length
        assert len(returns_to_go) == self.context_length
        assert len(mask) == self.context_length
        
        # Подготовка входов и целей
        # input_actions: [START, act_0, act_1, ..., act_{n-1}]
        # target_actions: [act_0, act_1, ..., act_n]
        
        input_actions = np.concatenate([
            np.array([self.action_dim], dtype=np.int64),  # START token
            actions[:-1]
        ], axis=0)
        
        target_actions = actions.copy()
        
        # ВАЖНО: Заменяем START токены (action_dim) в targets на -100
        # -100 автоматически игнорируется в CrossEntropyLoss
        target_actions[target_actions == self.action_dim] = -100
        
        return {
            'observations': torch.FloatTensor(observations),
            'actions': torch.LongTensor(input_actions),
            'returns_to_go': torch.FloatTensor(returns_to_go),  # НОВОЕ!
            'targets': torch.LongTensor(target_actions),
            'mask': torch.FloatTensor(mask)
        }


def prepare_gym_data(
    env_preset: str = 'cartpole',
    num_episodes: int = 1000,
    max_episode_length: int = 500,
    save_path: str = 'data/gym_trajectories.pkl',
    seed: int = 42,
    **env_kwargs
):
    """
    Собирает траектории из Gymnasium среды и сохраняет их
    
    Args:
        env_preset: название preset среды ('cartpole', 'lunar_lander', и т.д.)
        num_episodes: количество эпизодов для сбора
        max_episode_length: максимальная длина эпизода
        save_path: путь для сохранения данных
        seed: random seed
        **env_kwargs: дополнительные параметры для среды
        
    Returns:
        trajectories: list of trajectories
        env_config: dict с конфигурацией среды
    """
    print(f"\n{'='*80}")
    print(f"ПОДГОТОВКА ДАННЫХ ИЗ GYMNASIUM")
    print(f"{'='*80}")
    print(f"Среда: {env_preset}")
    print(f"Эпизодов: {num_episodes}")
    print(f"Макс длина: {max_episode_length}")
    
    # Создаем среду
    env = create_gym_environment(env_preset, seed=seed, **env_kwargs)
    
    # Сохраняем конфигурацию
    env_config = {
        'env_preset': env_preset,
        'env_name': env.env_name,
        'obs_dim': env.obs_dim,
        'action_dim': env.action_dim,
        'seed': seed
    }
    
    # Собираем траектории
    collector = GymnasiumTrajectoryCollector(env, max_episode_length=max_episode_length)
    trajectories = collector.collect_random_trajectories(num_episodes, verbose=True)
    
    # Сохраняем
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = {
        'trajectories': trajectories,
        'env_config': env_config
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Данные сохранены в {save_path}")
    
    env.close()
    
    return trajectories, env_config


def load_gym_trajectories(path='data/gym_trajectories.pkl'):
    """
    Загружает траектории и конфигурацию среды из файла
    
    Returns:
        trajectories: list of trajectories
        env_config: dict с конфигурацией среды
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    env_config = data.get('env_config', {})
    
    print(f"✓ Загружено траекторий: {len(trajectories)}")
    if env_config:
        print(f"  Среда: {env_config.get('env_name', 'unknown')}")
        print(f"  Obs dim: {env_config.get('obs_dim', '?')}")
        print(f"  Action dim: {env_config.get('action_dim', '?')}")
    
    return trajectories, env_config


def create_dataloaders(
    trajectories,
    context_length=20,
    action_dim=5,
    batch_size=32,
    train_split=0.9,
    gamma=1.0,
    num_workers=4,
    pin_memory=True,
    persistent_workers=None,
    prefetch_factor=None
):
    """
    Создает DataLoader'ы для обучения и валидации
    
    Args:
        trajectories: список траекторий
        context_length: длина контекстного окна
        action_dim: размерность действий
        batch_size: размер батча
        train_split: доля данных для обучения
        gamma: discount factor для returns-to-go
        num_workers: количество воркеров для загрузки данных
        pin_memory: использовать pinned memory для GPU
        persistent_workers: не убивать workers между эпохами
        prefetch_factor: количество батчей для предзагрузки
        
    Returns:
        train_loader, val_loader
    """
    # Разделяем на train и val
    split_idx = int(len(trajectories) * train_split)
    train_trajectories = trajectories[:split_idx]
    val_trajectories = trajectories[split_idx:]
    
    print(f"\nСоздание DataLoaders:")
    print(f"  Train траекторий: {len(train_trajectories)}")
    print(f"  Val траекторий: {len(val_trajectories)}")
    print(f"  Gamma (discount): {gamma}")
    
    # Создаем датасеты
    train_dataset = RLDataset(train_trajectories, context_length, action_dim, gamma)
    val_dataset = RLDataset(val_trajectories, context_length, action_dim, gamma)
    
    print(f"  Train сэмплов: {len(train_dataset)}")
    print(f"  Val сэмплов: {len(val_dataset)}")
    
    # Параметры для DataLoader
    # persistent_workers и prefetch_factor работают только с num_workers > 0
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    if num_workers > 0:
        if persistent_workers is None:
            persistent_workers = True
        if prefetch_factor is None:
            prefetch_factor = 2
        
        dataloader_kwargs['persistent_workers'] = persistent_workers
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    # Создаем dataloaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    
    return train_loader, val_loader


def prepare_multiple_envs(
    env_presets=['cartpole', 'mountain_car'],
    num_episodes_per_env=500,
    save_dir='data/multi_env'
):
    """
    Собирает данные из нескольких сред
    
    Args:
        env_presets: список preset сред
        num_episodes_per_env: количество эпизодов для каждой среды
        save_dir: директория для сохранения
        
    Returns:
        all_trajectories: объединенные траектории
        env_configs: конфигурации всех сред
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_trajectories = []
    env_configs = []
    
    print(f"\n{'='*80}")
    print(f"СБОР ДАННЫХ ИЗ НЕСКОЛЬКИХ СРЕД")
    print(f"{'='*80}")
    
    for preset in env_presets:
        save_path = os.path.join(save_dir, f'{preset}_trajectories.pkl')
        
        trajectories, env_config = prepare_gym_data(
            env_preset=preset,
            num_episodes=num_episodes_per_env,
            save_path=save_path
        )
        
        all_trajectories.extend(trajectories)
        env_configs.append(env_config)
        
        print(f"\n{'='*80}\n")
    
    # Сохраняем объединенные данные
    combined_path = os.path.join(save_dir, 'combined_trajectories.pkl')
    data = {
        'trajectories': all_trajectories,
        'env_configs': env_configs
    }
    
    with open(combined_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Объединенные данные сохранены: {combined_path}")
    print(f"  Всего траекторий: {len(all_trajectories)}")
    print(f"  Из {len(env_configs)} сред")
    
    return all_trajectories, env_configs


if __name__ == '__main__':
    # Пример 1: Подготовка данных из CartPole
    print("\n" + "="*80)
    print("ПРИМЕР 1: ПОДГОТОВКА ДАННЫХ ИЗ CARTPOLE")
    print("="*80)
    
    trajectories, env_config = prepare_gym_data(
        env_preset='cartpole',
        num_episodes=100,
        max_episode_length=200,
        save_path='data/cartpole_trajectories.pkl',
        seed=42
    )
    
    # Пример 2: Создание DataLoaders с RTG
    print("\n" + "="*80)
    print("ПРИМЕР 2: СОЗДАНИЕ DATALOADERS С RETURNS-TO-GO")
    print("="*80)
    
    train_loader, val_loader = create_dataloaders(
        trajectories,
        context_length=20,
        action_dim=env_config['action_dim'],
        batch_size=32,
        gamma=1.0  # Недисконтированные returns
    )
    
    # Проверяем батч
    batch = next(iter(train_loader))
    print(f"\nПример батча:")
    print(f"  Observations: {batch['observations'].shape}")
    print(f"  Actions: {batch['actions'].shape}")
    print(f"  Returns-to-go: {batch['returns_to_go'].shape}")  # НОВОЕ!
    print(f"  Targets: {batch['targets'].shape}")
    print(f"  Mask: {batch['mask'].shape}")
    
    # Проверяем статистику RTG
    print(f"\nСтатистика Returns-to-go в батче:")
    print(f"  Min: {batch['returns_to_go'].min():.2f}")
    print(f"  Max: {batch['returns_to_go'].max():.2f}")
    print(f"  Mean: {batch['returns_to_go'].mean():.2f}")
    print(f"  Std: {batch['returns_to_go'].std():.2f}")
    
    # Пример 3: Доступные среды
    print("\n" + "="*80)
    print("ПРИМЕР 3: ДОСТУПНЫЕ PRESET СРЕДЫ")
    print("="*80)
    
    for preset_name, config in PRESET_ENVS.items():
        print(f"\n{preset_name}:")
        print(f"  Environment: {config['env_name']}")
        print(f"  Observation dim: {config['obs_dim']}")
        print(f"  Action dim: {config['action_dim']}")
        print(f"  Max length: {config['max_episode_length']}")
        print(f"  Описание: {config['description']}")
    
    print("\n" + "="*80)
    print("Для использования выберите preset:")
    print("  prepare_gym_data(env_preset='cartpole', num_episodes=1000)")
    print("  prepare_gym_data(env_preset='lunar_lander', num_episodes=1000)")
    print("  и т.д.")
    print("="*80)