"""
Gymnasium Environment Wrapper для Decision Transformer
Поддерживает различные типы observation и action spaces
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional


class GymnasiumWrapper:
    """
    Обертка для Gymnasium сред
    Приводит наблюдения и действия к нужному формату для Decision Transformer
    """
    
    def __init__(
        self,
        env_name: str,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Args:
            env_name: название среды (например, 'CartPole-v1')
            obs_dim: целевая размерность наблюдений (если нужно преобразование)
            action_dim: целевая размерность действий (если нужно преобразование)
            seed: random seed
            render_mode: режим рендеринга ('human', 'rgb_array', None)
        """
        self.env = gym.make(env_name, render_mode=render_mode)
        self.env_name = env_name
        self.seed = seed
        
        if seed is not None:
            self.env.reset(seed=seed)
        
        # Определяем размерности из среды
        self._setup_spaces()
        
        # Если заданы целевые размерности, используем их
        self.target_obs_dim = obs_dim if obs_dim is not None else self.native_obs_dim
        self.target_action_dim = action_dim if action_dim is not None else self.native_action_dim
        
        print(f"Среда: {env_name}")
        print(f"  Observation space: {self.env.observation_space}")
        print(f"  Action space: {self.env.action_space}")
        print(f"  Native obs_dim: {self.native_obs_dim}")
        print(f"  Native action_dim: {self.native_action_dim}")
        print(f"  Target obs_dim: {self.target_obs_dim}")
        print(f"  Target action_dim: {self.target_action_dim}")
    
    def _setup_spaces(self):
        """Определяет размерности observation и action spaces"""
        obs_space = self.env.observation_space
        action_space = self.env.action_space
        
        # Observation space
        if isinstance(obs_space, gym.spaces.Box):
            self.obs_type = 'continuous'
            self.native_obs_dim = int(np.prod(obs_space.shape))
        elif isinstance(obs_space, gym.spaces.Discrete):
            self.obs_type = 'discrete'
            self.native_obs_dim = obs_space.n
        elif isinstance(obs_space, gym.spaces.MultiDiscrete):
            self.obs_type = 'multi_discrete'
            self.native_obs_dim = int(np.sum(obs_space.nvec))
        else:
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")
        
        # Action space
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_type = 'discrete'
            self.native_action_dim = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            self.action_type = 'continuous'
            self.native_action_dim = int(np.prod(action_space.shape))
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.action_type = 'multi_discrete'
            self.native_action_dim = int(np.sum(action_space.nvec))
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")
    
    def _process_observation(self, obs: Any) -> np.ndarray:
        """
        Преобразует наблюдение в вектор нужной размерности
        
        Returns:
            observation: np.array [obs_dim]
        """
        if self.obs_type == 'continuous':
            # Flatten если многомерное
            obs_flat = obs.flatten().astype(np.float32)
        elif self.obs_type == 'discrete':
            # One-hot encoding для дискретных наблюдений
            obs_flat = np.zeros(self.native_obs_dim, dtype=np.float32)
            obs_flat[obs] = 1.0
        elif self.obs_type == 'multi_discrete':
            # One-hot для каждого компонента
            obs_flat = []
            offset = 0
            for i, n in enumerate(self.env.observation_space.nvec):
                one_hot = np.zeros(n, dtype=np.float32)
                one_hot[obs[i]] = 1.0
                obs_flat.append(one_hot)
            obs_flat = np.concatenate(obs_flat)
        
        # Приводим к целевой размерности
        if len(obs_flat) < self.target_obs_dim:
            # Padding нулями
            obs_flat = np.pad(obs_flat, (0, self.target_obs_dim - len(obs_flat)))
        elif len(obs_flat) > self.target_obs_dim:
            # Truncate или compress
            obs_flat = obs_flat[:self.target_obs_dim]
        
        return obs_flat
    
    def _process_action(self, action: int) -> Any:
        """
        Преобразует action из формата Decision Transformer в формат среды
        
        Args:
            action: int от 0 до target_action_dim-1
            
        Returns:
            action в формате среды
        """
        if self.action_type == 'discrete':
            # Для дискретных действий - просто возвращаем индекс
            return int(action % self.native_action_dim)
        elif self.action_type == 'continuous':
            # Для непрерывных действий - преобразуем индекс в значение
            # Дискретизация непрерывного пространства
            action_space = self.env.action_space
            low = action_space.low
            high = action_space.high
            
            # Преобразуем дискретное действие в непрерывное
            num_bins = self.target_action_dim
            bin_idx = action % num_bins
            
            # Линейная интерполяция
            action_continuous = low + (high - low) * (bin_idx / (num_bins - 1))
            return action_continuous.astype(np.float32)
        elif self.action_type == 'multi_discrete':
            # Для multi-discrete - декодируем индекс
            nvec = self.env.action_space.nvec
            action_multi = []
            remaining = action
            for n in reversed(nvec):
                action_multi.append(remaining % n)
                remaining //= n
            return np.array(list(reversed(action_multi)))
        
        return action
    
    def reset(self) -> np.ndarray:
        """
        Сброс среды
        
        Returns:
            observation: np.array [obs_dim]
        """
        obs, info = self.env.reset()
        return self._process_observation(obs)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Выполнение действия
        
        Args:
            action: int от 0 до action_dim-1
            
        Returns:
            observation: np.array [obs_dim]
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        processed_action = self._process_action(action)
        obs, reward, terminated, truncated, info = self.env.step(processed_action)
        obs_processed = self._process_observation(obs)
        
        # Объединяем terminated и truncated для совместимости
        done = terminated or truncated
        
        return obs_processed, float(reward), done, info
    
    def sample_action(self) -> int:
        """
        Сэмплирование случайного действия
        
        Returns:
            action: int от 0 до action_dim-1
        """
        return np.random.randint(0, self.target_action_dim)
    
    def close(self):
        """Закрывает среду"""
        self.env.close()
    
    @property
    def obs_dim(self):
        return self.target_obs_dim
    
    @property
    def action_dim(self):
        return self.target_action_dim


class GymnasiumTrajectoryCollector:
    """
    Сборщик траекторий из Gymnasium сред
    """
    
    def __init__(
        self,
        env: GymnasiumWrapper,
        max_episode_length: int = 1000
    ):
        self.env = env
        self.max_episode_length = max_episode_length
    
    def collect_random_trajectories(
        self,
        num_episodes: int,
        verbose: bool = True
    ) -> list:
        """
        Собирает траектории со случайной политикой
        
        Args:
            num_episodes: количество эпизодов
            verbose: выводить прогресс
            
        Returns:
            trajectories: list of dicts с ключами:
                'observations': list of np.arrays
                'actions': list of ints
                'rewards': list of floats
        """
        trajectories = []
        total_reward = 0
        total_steps = 0
        
        if verbose:
            print(f"\nСбор {num_episodes} траекторий из {self.env.env_name}...")
        
        for episode in range(num_episodes):
            obs_list = []
            action_list = []
            reward_list = []
            
            obs = self.env.reset()
            obs_list.append(obs)
            
            episode_reward = 0
            
            for step in range(self.max_episode_length):
                # Случайное действие
                action = self.env.sample_action()
                
                obs, reward, done, info = self.env.step(action)
                
                action_list.append(action)
                reward_list.append(reward)
                obs_list.append(obs)
                
                episode_reward += reward
                
                if done:
                    break
            
            trajectories.append({
                'observations': obs_list,
                'actions': action_list,
                'rewards': reward_list
            })
            
            total_reward += episode_reward
            total_steps += len(action_list)
            
            if verbose and (episode + 1) % max(1, num_episodes // 10) == 0:
                avg_reward = total_reward / (episode + 1)
                avg_length = total_steps / (episode + 1)
                print(f"  Эпизод {episode + 1}/{num_episodes}: "
                      f"avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}")
        
        if verbose:
            print(f"\n✓ Собрано траекторий: {len(trajectories)}")
            print(f"  Всего шагов: {total_steps}")
            print(f"  Средняя длина: {total_steps / num_episodes:.2f}")
            print(f"  Средняя награда: {total_reward / num_episodes:.2f}")
        
        return trajectories


# Предустановленные конфигурации для популярных сред
PRESET_ENVS = {
    'cartpole': {
        'env_name': 'CartPole-v1',
        'obs_dim': 4,
        'action_dim': 2,
        'max_episode_length': 500,
        'description': 'Классическая задача удержания шеста'
    },
    'lunar_lander': {
        'env_name': 'LunarLander-v2',
        'obs_dim': 8,
        'action_dim': 4,
        'max_episode_length': 1000,
        'description': 'Посадка лунного модуля'
    },
    'mountain_car': {
        'env_name': 'MountainCar-v0',
        'obs_dim': 2,
        'action_dim': 3,
        'max_episode_length': 200,
        'description': 'Машинка должна выбраться из долины'
    },
    'acrobot': {
        'env_name': 'Acrobot-v1',
        'obs_dim': 6,
        'action_dim': 3,
        'max_episode_length': 500,
        'description': 'Двухзвенный маятник'
    },
    'pendulum': {
        'env_name': 'Pendulum-v1',
        'obs_dim': 3,
        'action_dim': 5,  # Дискретизация непрерывного действия
        'max_episode_length': 200,
        'description': 'Перевернутый маятник (continuous control)'
    }
}


def create_gym_environment(preset: str = 'cartpole', **kwargs) -> GymnasiumWrapper:
    """
    Создает Gymnasium среду из preset конфигурации
    
    Args:
        preset: название preset ('cartpole', 'lunar_lander', и т.д.)
        **kwargs: дополнительные параметры для GymnasiumWrapper
        
    Returns:
        env: GymnasiumWrapper
    """
    if preset not in PRESET_ENVS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_ENVS.keys())}")
    
    config = PRESET_ENVS[preset].copy()
    config.update(kwargs)
    
    print(f"\n{'='*80}")
    print(f"Создание среды: {preset}")
    print(f"Описание: {config['description']}")
    print(f"{'='*80}")
    
    env = GymnasiumWrapper(
        env_name=config['env_name'],
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        seed=config.get('seed')
    )
    
    return env


if __name__ == '__main__':
    # Пример использования
    print("="*80)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ GYMNASIUM WRAPPER")
    print("="*80)
    
    # Пример 1: CartPole
    print("\n--- Пример 1: CartPole-v1 ---")
    env = create_gym_environment('cartpole', seed=42)
    
    obs = env.reset()
    print(f"Начальное наблюдение: {obs.shape}")
    
    for i in range(5):
        action = env.sample_action()
        obs, reward, done, info = env.step(action)
        print(f"Шаг {i+1}: action={action}, reward={reward:.2f}, done={done}")
        if done:
            break
    
    env.close()
    
    # Пример 2: Сбор траекторий
    print("\n--- Пример 2: Сбор траекторий ---")
    env = create_gym_environment('cartpole', seed=42)
    collector = GymnasiumTrajectoryCollector(env, max_episode_length=200)
    
    trajectories = collector.collect_random_trajectories(num_episodes=5)
    
    print(f"\nПервая траектория:")
    print(f"  Длина: {len(trajectories[0]['actions'])}")
    print(f"  Суммарная награда: {sum(trajectories[0]['rewards']):.2f}")
    
    env.close()
    
    # Пример 3: Разные среды
    print("\n--- Пример 3: Доступные preset среды ---")
    for preset_name, config in PRESET_ENVS.items():
        print(f"\n{preset_name}:")
        print(f"  Env: {config['env_name']}")
        print(f"  Obs dim: {config['obs_dim']}, Action dim: {config['action_dim']}")
        print(f"  Описание: {config['description']}")
