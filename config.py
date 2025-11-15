"""
Конфигурационный файл для Decision Transformer с MoE и GQA

Используйте этот файл для настройки всех параметров обучения.
Импортируйте get_config() в ваших скриптах для получения настроек.
"""

import torch


def get_config(mode='default'):
    """
    Возвращает конфигурацию для обучения
    
    Args:
        mode: 'default', 'quick_test', 'large', 'small'
        
    Returns:
        config: dict с параметрами
    """
    
    # Базовая конфигурация
    base_config = {
        # Среда
        'obs_dim': 1000,
        'action_dim': 5,
        'max_episode_length': 100,
        
        # Данные
        'num_episodes': 1000,
        'context_length': 20,
        'train_split': 0.9,
        
        # Модель - Трансформер
        'embed_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'num_kv_heads': 4,      # GQA: половина от num_heads
        'num_experts': 8,        # MoE: количество экспертов
        'expert_top_k': 2,       # MoE: сколько экспертов активировать
        'max_seq_len': 512,
        'dropout': 0.1,
        
        # Обучение
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        
        # Логирование
        'log_interval': 10,
        'save_interval': 5,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        
        # Система
        'num_workers': 4,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Режимы
    if mode == 'quick_test':
        # Быстрый тест (для отладки)
        config = base_config.copy()
        config.update({
            'num_episodes': 50,
            'num_epochs': 3,
            'batch_size': 8,
            'embed_dim': 128,
            'num_layers': 2,
            'num_experts': 4,
            'save_interval': 1,
        })
        return config
    
    elif mode == 'small':
        # Маленькая модель (быстрое обучение)
        config = base_config.copy()
        config.update({
            'embed_dim': 128,
            'num_layers': 4,
            'num_heads': 4,
            'num_kv_heads': 2,
            'num_experts': 4,
            'batch_size': 64,
        })
        return config
    
    elif mode == 'large':
        # Большая модель (лучшее качество)
        config = base_config.copy()
        config.update({
            'num_episodes': 5000,
            'embed_dim': 512,
            'num_layers': 12,
            'num_heads': 16,
            'num_kv_heads': 8,
            'num_experts': 16,
            'batch_size': 16,
            'num_epochs': 100,
            'learning_rate': 1e-4,
        })
        return config
    
    elif mode == 'huge':
        # Огромная модель (для мощных GPU)
        config = base_config.copy()
        config.update({
            'num_episodes': 10000,
            'embed_dim': 1024,
            'num_layers': 24,
            'num_heads': 32,
            'num_kv_heads': 16,
            'num_experts': 32,
            'batch_size': 8,
            'num_epochs': 200,
            'learning_rate': 5e-5,
            'context_length': 50,
        })
        return config
    
    elif mode == 'default':
        return base_config
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: default, quick_test, small, large, huge")


def get_custom_config(**kwargs):
    """
    Создает кастомную конфигурацию
    
    Usage:
        config = get_custom_config(
            obs_dim=2000,
            action_dim=10,
            embed_dim=512,
            num_layers=8
        )
    """
    config = get_config('default')
    config.update(kwargs)
    return config


def print_config(config):
    """Красиво выводит конфигурацию"""
    print("\n" + "="*80)
    print("КОНФИГУРАЦИЯ")
    print("="*80)
    
    sections = {
        'Среда': ['obs_dim', 'action_dim', 'max_episode_length'],
        'Данные': ['num_episodes', 'context_length', 'train_split'],
        'Модель': ['embed_dim', 'num_layers', 'num_heads', 'num_kv_heads', 
                   'num_experts', 'expert_top_k', 'max_seq_len', 'dropout'],
        'Обучение': ['batch_size', 'num_epochs', 'learning_rate', 
                     'weight_decay', 'grad_clip'],
        'Система': ['device', 'num_workers', 'seed']
    }
    
    for section_name, keys in sections.items():
        print(f"\n{section_name}:")
        for key in keys:
            if key in config:
                print(f"  {key:20s}: {config[key]}")
    
    print("\n" + "="*80)


def estimate_memory(config):
    """
    Оценивает требования к памяти GPU
    
    Returns:
        estimated_gb: примерная память в GB
    """
    # Упрощенная оценка памяти для модели
    embed_dim = config['embed_dim']
    num_layers = config['num_layers']
    num_experts = config['num_experts']
    obs_dim = config['obs_dim']
    action_dim = config['action_dim']
    
    # Embeddings
    memory = obs_dim * embed_dim  # obs encoder
    memory += action_dim * embed_dim  # action embedding
    memory += config['max_seq_len'] * embed_dim  # positional
    
    # Трансформер блоки
    for _ in range(num_layers):
        # Attention
        memory += 3 * embed_dim * embed_dim  # Q, K, V projections (упрощенно)
        memory += embed_dim * embed_dim  # output projection
        
        # MoE
        expert_dim = 4 * embed_dim
        memory += num_experts * (embed_dim * expert_dim + expert_dim * embed_dim)
        memory += embed_dim * num_experts  # router
    
    # Output head
    memory += embed_dim * action_dim
    
    # Конвертируем в GB (float32 = 4 bytes)
    # Умножаем на 4 для учета градиентов, оптимизатора и активаций
    estimated_gb = (memory * 4 * 4) / (1024**3)
    
    return estimated_gb


def recommend_config(gpu_memory_gb):
    """
    Рекомендует конфигурацию на основе доступной памяти GPU
    
    Args:
        gpu_memory_gb: доступная память GPU в GB
        
    Returns:
        recommended_mode: строка с рекомендуемым режимом
    """
    if gpu_memory_gb < 4:
        return 'small'
    elif gpu_memory_gb < 8:
        return 'default'
    elif gpu_memory_gb < 16:
        return 'large'
    else:
        return 'huge'


# Предустановленные конфигурации для разных задач
PRESET_CONFIGS = {
    'atari': {
        'obs_dim': 128 * 128,  # Сжатое изображение
        'action_dim': 18,       # Стандартный набор действий Atari
        'embed_dim': 512,
        'num_layers': 8,
        'context_length': 30,
    },
    
    'robotics': {
        'obs_dim': 50,          # Состояния робота (позиции, скорости, и т.д.)
        'action_dim': 7,        # 7-DOF манипулятор
        'embed_dim': 256,
        'num_layers': 6,
        'context_length': 50,   # Длинная история для точного контроля
    },
    
    'text_gen': {
        'obs_dim': 768,         # BERT embeddings
        'action_dim': 30000,    # Размер словаря
        'embed_dim': 1024,
        'num_layers': 12,
        'context_length': 100,
    },
    
    'trading': {
        'obs_dim': 200,         # Финансовые индикаторы
        'action_dim': 3,        # Buy/Hold/Sell
        'embed_dim': 256,
        'num_layers': 8,
        'context_length': 100,  # Длинная история для анализа трендов
    }
}


def get_preset_config(preset_name):
    """
    Получить предустановленную конфигурацию для конкретной задачи
    
    Args:
        preset_name: 'atari', 'robotics', 'text_gen', 'trading'
        
    Returns:
        config: dict
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    config = get_config('default')
    config.update(PRESET_CONFIGS[preset_name])
    
    return config


if __name__ == '__main__':
    # Примеры использования
    
    print("\n1. Конфигурация по умолчанию:")
    config = get_config('default')
    print_config(config)
    
    print("\n2. Оценка памяти:")
    mem = estimate_memory(config)
    print(f"Примерная требуемая память GPU: {mem:.2f} GB")
    
    print("\n3. Быстрый тест конфигурация:")
    test_config = get_config('quick_test')
    print_config(test_config)
    
    print("\n4. Рекомендации по GPU:")
    for gpu_mem in [4, 8, 16, 24]:
        mode = recommend_config(gpu_mem)
        print(f"  {gpu_mem} GB GPU -> режим: '{mode}'")
    
    print("\n5. Preset конфигурации:")
    for preset in PRESET_CONFIGS.keys():
        config = get_preset_config(preset)
        print(f"\n  {preset}:")
        print(f"    obs_dim: {config['obs_dim']}, action_dim: {config['action_dim']}")
        print(f"    context_length: {config['context_length']}")
