"""
Параллельный сбор данных из Gymnasium сред с Ray
Ускорение сбора данных в 10-16x на многоядерных системах
"""

import ray
import numpy as np
import pickle
import os
from typing import List, Dict
from gym_environment import GymnasiumWrapper, create_gym_environment


@ray.remote
class ParallelEnvRunner:
    """
    Ray worker для параллельного сбора траекторий из среды
    """
    
    def __init__(self, env_preset: str, seed: int = None, **env_kwargs):
        """
        Args:
            env_preset: название preset среды
            seed: random seed для этого worker
            **env_kwargs: дополнительные параметры для среды
        """
        self.env = create_gym_environment(env_preset, seed=seed, **env_kwargs)
        self.env_preset = env_preset
        self.seed = seed
    
    def collect_episodes(self, num_episodes: int, max_episode_length: int = 1000) -> List[Dict]:
        """
        Собирает эпизоды со случайной политикой
        
        Args:
            num_episodes: количество эпизодов для сбора
            max_episode_length: максимальная длина эпизода
            
        Returns:
            episodes: список собранных эпизодов
        """
        episodes = []
        
        for _ in range(num_episodes):
            obs_list = []
            action_list = []
            reward_list = []
            
            obs = self.env.reset()
            obs_list.append(obs)
            
            for step in range(max_episode_length):
                action = self.env.sample_action()
                obs, reward, done, info = self.env.step(action)
                
                action_list.append(action)
                reward_list.append(reward)
                obs_list.append(obs)
                
                if done:
                    break
            
            episodes.append({
                'observations': obs_list,
                'actions': action_list,
                'rewards': reward_list
            })
        
        return episodes
    
    def collect_episodes_with_policy(
        self, 
        num_episodes: int,
        policy_fn,
        max_episode_length: int = 1000
    ) -> List[Dict]:
        """
        Собирает эпизоды с заданной политикой (для экспертных траекторий)
        
        Args:
            num_episodes: количество эпизодов
            policy_fn: функция политики (obs) -> action
            max_episode_length: максимальная длина эпизода
            
        Returns:
            episodes: список собранных эпизодов
        """
        episodes = []
        
        for _ in range(num_episodes):
            obs_list = []
            action_list = []
            reward_list = []
            
            obs = self.env.reset()
            obs_list.append(obs)
            
            for step in range(max_episode_length):
                # Используем заданную политику
                action = policy_fn(obs)
                obs, reward, done, info = self.env.step(action)
                
                action_list.append(action)
                reward_list.append(reward)
                obs_list.append(obs)
                
                if done:
                    break
            
            episodes.append({
                'observations': obs_list,
                'actions': action_list,
                'rewards': reward_list
            })
        
        return episodes
    
    def get_stats(self) -> Dict:
        """Возвращает информацию о среде"""
        return {
            'env_preset': self.env_preset,
            'obs_dim': self.env.obs_dim,
            'action_dim': self.env.action_dim,
            'seed': self.seed
        }


def parallel_collect_data(
    env_preset: str = 'cartpole',
    num_episodes: int = 1000,
    max_episode_length: int = 500,
    num_workers: int = None,
    save_path: str = 'data/gym_trajectories.pkl',
    seed: int = 42,
    verbose: bool = True,
    **env_kwargs
):
    """
    Параллельный сбор траекторий с Ray
    
    Args:
        env_preset: название preset среды
        num_episodes: общее количество эпизодов
        max_episode_length: максимальная длина эпизода
        num_workers: количество параллельных workers (по умолчанию - количество CPU)
        save_path: путь для сохранения
        seed: базовый random seed
        verbose: выводить прогресс
        **env_kwargs: дополнительные параметры для среды
        
    Returns:
        trajectories: список траекторий
        env_config: конфигурация среды
    """
    import time
    
    # Инициализируем Ray если еще не запущен
    if not ray.is_initialized():
        # Определяем количество CPU
        import multiprocessing
        num_cpus = multiprocessing.cpu_count()
        
        if num_workers is None:
            num_workers = min(num_cpus, 16)  # Ограничиваем 16 workers
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ИНИЦИАЛИЗАЦИЯ RAY")
            print(f"{'='*80}")
            print(f"Доступно CPU: {num_cpus}")
            print(f"Используем workers: {num_workers}")
        
        ray.init(num_cpus=num_workers, ignore_reinit_error=True)
    else:
        if num_workers is None:
            num_workers = min(ray.cluster_resources().get('CPU', 1), 16)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"ПАРАЛЛЕЛЬНЫЙ СБОР ДАННЫХ")
        print(f"{'='*80}")
        print(f"Среда: {env_preset}")
        print(f"Эпизодов: {num_episodes}")
        print(f"Workers: {num_workers}")
        print(f"Эпизодов на worker: ~{num_episodes // num_workers}")
    
    start_time = time.time()
    
    # Создаем workers с разными seeds
    workers = []
    for i in range(num_workers):
        worker_seed = seed + i if seed is not None else None
        worker = ParallelEnvRunner.remote(
            env_preset=env_preset,
            seed=worker_seed,
            **env_kwargs
        )
        workers.append(worker)
    
    # Получаем конфигурацию среды от первого worker
    env_config = ray.get(workers[0].get_stats.remote())
    env_config['env_preset'] = env_preset
    env_config['seed'] = seed
    
    if verbose:
        print(f"\nКонфигурация среды:")
        print(f"  Observation dim: {env_config['obs_dim']}")
        print(f"  Action dim: {env_config['action_dim']}")
    
    # Распределяем эпизоды между workers
    episodes_per_worker = num_episodes // num_workers
    remaining_episodes = num_episodes % num_workers
    
    # Запускаем параллельный сбор
    if verbose:
        print(f"\n🚀 Запускаем параллельный сбор...")
    
    futures = []
    for i, worker in enumerate(workers):
        # Последнему worker даем оставшиеся эпизоды
        worker_episodes = episodes_per_worker + (remaining_episodes if i == len(workers) - 1 else 0)
        
        future = worker.collect_episodes.remote(
            num_episodes=worker_episodes,
            max_episode_length=max_episode_length
        )
        futures.append(future)
    
    # Собираем результаты
    if verbose:
        print(f"Ожидаем завершения workers...")
    
    all_episodes = []
    completed_workers = 0
    
    # Можно использовать ray.wait для отслеживания прогресса
    while futures:
        done_futures, futures = ray.wait(futures, num_returns=1)
        episodes = ray.get(done_futures[0])
        all_episodes.extend(episodes)
        completed_workers += 1
        
        if verbose:
            print(f"  Worker {completed_workers}/{num_workers} завершен "
                  f"({len(all_episodes)}/{num_episodes} эпизодов)")
    
    collection_time = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Сбор завершен за {collection_time:.2f} секунд")
        
        # Статистика
        total_steps = sum(len(ep['actions']) for ep in all_episodes)
        total_reward = sum(sum(ep['rewards']) for ep in all_episodes)
        avg_length = total_steps / len(all_episodes)
        avg_reward = total_reward / len(all_episodes)
        
        print(f"\nСтатистика:")
        print(f"  Собрано эпизодов: {len(all_episodes)}")
        print(f"  Всего шагов: {total_steps}")
        print(f"  Средняя длина: {avg_length:.2f}")
        print(f"  Средняя награда: {avg_reward:.2f}")
        print(f"  Скорость: {len(all_episodes) / collection_time:.2f} эпизодов/сек")
    
    # Сохраняем
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            'trajectories': all_episodes,
            'env_config': env_config,
            'collection_time': collection_time,
            'num_workers': num_workers
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        if verbose:
            print(f"\n✓ Данные сохранены: {save_path}")
    
    return all_episodes, env_config


def parallel_collect_multiple_envs(
    env_presets: List[str],
    num_episodes_per_env: int = 500,
    max_episode_length: int = 500,
    num_workers: int = None,
    save_dir: str = 'data/multi_env',
    seed: int = 42,
    verbose: bool = True
):
    """
    Параллельный сбор данных из нескольких сред одновременно
    
    Args:
        env_presets: список preset сред
        num_episodes_per_env: эпизодов для каждой среды
        max_episode_length: максимальная длина эпизода
        num_workers: количество workers (распределяется между средами)
        save_dir: директория для сохранения
        seed: базовый seed
        verbose: выводить прогресс
        
    Returns:
        all_trajectories: объединенные траектории
        env_configs: конфигурации всех сред
    """
    if not ray.is_initialized():
        import multiprocessing
        total_cpus = multiprocessing.cpu_count()
        if num_workers is None:
            num_workers = min(total_cpus, 16)
        ray.init(num_cpus=num_workers, ignore_reinit_error=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"ПАРАЛЛЕЛЬНЫЙ СБОР ИЗ НЕСКОЛЬКИХ СРЕД")
        print(f"{'='*80}")
        print(f"Среды: {env_presets}")
        print(f"Эпизодов на среду: {num_episodes_per_env}")
        print(f"Workers: {num_workers}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Запускаем параллельный сбор для всех сред одновременно
    futures = []
    for i, preset in enumerate(env_presets):
        save_path = os.path.join(save_dir, f'{preset}_trajectories.pkl')
        
        # Каждая среда получает свою долю workers
        workers_per_env = max(1, num_workers // len(env_presets))
        
        # Запускаем асинхронно
        future = ray.remote(parallel_collect_data).remote(
            env_preset=preset,
            num_episodes=num_episodes_per_env,
            max_episode_length=max_episode_length,
            num_workers=workers_per_env,
            save_path=save_path,
            seed=seed + i * 1000,
            verbose=False  # Отключаем verbose для каждой среды
        )
        futures.append((preset, future))
    
    # Собираем результаты
    all_trajectories = []
    env_configs = []
    
    for preset, future in futures:
        if verbose:
            print(f"\nОжидаем {preset}...")
        
        trajectories, env_config = ray.get(future)
        all_trajectories.extend(trajectories)
        env_configs.append(env_config)
        
        if verbose:
            print(f"✓ {preset}: {len(trajectories)} эпизодов")
    
    # Сохраняем объединенные данные
    combined_path = os.path.join(save_dir, 'combined_trajectories.pkl')
    data = {
        'trajectories': all_trajectories,
        'env_configs': env_configs
    }
    
    with open(combined_path, 'wb') as f:
        pickle.dump(data, f)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"✓ Объединенные данные: {combined_path}")
        print(f"  Всего траекторий: {len(all_trajectories)}")
        print(f"  Из {len(env_configs)} сред")
        print(f"{'='*80}")
    
    return all_trajectories, env_configs


def benchmark_parallel_vs_sequential():
    """
    Сравнение параллельного и последовательного сбора
    """
    import time
    from gym_environment import create_gym_environment, GymnasiumTrajectoryCollector
    
    num_episodes = 100
    
    print(f"\n{'='*80}")
    print(f"БЕНЧМАРК: ПАРАЛЛЕЛЬНЫЙ vs ПОСЛЕДОВАТЕЛЬНЫЙ СБОР")
    print(f"{'='*80}")
    print(f"Эпизодов: {num_episodes}")
    
    # Последовательный сбор
    print(f"\n1. ПОСЛЕДОВАТЕЛЬНЫЙ СБОР:")
    start = time.time()
    env = create_gym_environment('cartpole', seed=42)
    collector = GymnasiumTrajectoryCollector(env, max_episode_length=500)
    trajectories_seq = collector.collect_random_trajectories(num_episodes, verbose=False)
    env.close()
    time_seq = time.time() - start
    print(f"   Время: {time_seq:.2f} секунд")
    
    # Параллельный сбор
    print(f"\n2. ПАРАЛЛЕЛЬНЫЙ СБОР (Ray):")
    start = time.time()
    trajectories_par, _ = parallel_collect_data(
        env_preset='cartpole',
        num_episodes=num_episodes,
        max_episode_length=500,
        seed=42,
        save_path=None,
        verbose=False
    )
    time_par = time.time() - start
    print(f"   Время: {time_par:.2f} секунд")
    
    # Результаты
    speedup = time_seq / time_par
    print(f"\n{'='*80}")
    print(f"РЕЗУЛЬТАТЫ:")
    print(f"  Последовательно: {time_seq:.2f}s")
    print(f"  Параллельно: {time_par:.2f}s")
    print(f"  Ускорение: {speedup:.2f}x")
    print(f"{'='*80}")
    
    # Очищаем Ray
    ray.shutdown()


if __name__ == '__main__':
    # Пример 1: Простой параллельный сбор
    print("\n" + "="*80)
    print("ПРИМЕР 1: ПАРАЛЛЕЛЬНЫЙ СБОР ДАННЫХ")
    print("="*80)
    
    trajectories, env_config = parallel_collect_data(
        env_preset='cartpole',
        num_episodes=100,
        max_episode_length=200,
        num_workers=4,
        save_path='data/parallel_cartpole.pkl',
        seed=42
    )
    
    # Пример 2: Бенчмарк
    print("\n" + "="*80)
    print("ПРИМЕР 2: БЕНЧМАРК")
    print("="*80)
    
    benchmark_parallel_vs_sequential()
    
    # Пример 3: Несколько сред одновременно
    print("\n" + "="*80)
    print("ПРИМЕР 3: МУЛЬТИ-СРЕДА")
    print("="*80)
    
    all_traj, all_configs = parallel_collect_multiple_envs(
        env_presets=['cartpole', 'mountain_car'],
        num_episodes_per_env=50,
        num_workers=8,
        save_dir='data/multi_env_parallel'
    )
    
    ray.shutdown()
    print("\n✓ Все примеры выполнены!")
