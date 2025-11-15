#!/usr/bin/env python3
"""
Запуск GRPO обучения (второй этап) с опциональной поддержкой Ray

Использование:
1. Сначала обучите модель на supervised learning (этап 1)
2. Затем загрузите эту модель и обучите с GRPO (этап 2)

Ray распараллеливание:
--use_ray        - использовать Ray для параллельного сбора роллаутов (6-8x ускорение)
--no_ray         - использовать последовательное выполнение (по умолчанию)
"""

import os
import sys
import argparse
import torch
from datetime import datetime

from model import DecisionTransformer

# Пытаемся импортировать Ray версию, если не получается - используем обычную
try:
    from train_grpo_ray import ValueNetwork, GRPOTrainer
    RAY_AVAILABLE = True
except ImportError:
    from train_grpo import ValueNetwork, GRPOTrainer
    RAY_AVAILABLE = False
    
from gym_environment import create_gym_environment


def run_grpo_training(
    pretrained_checkpoint,
    env_preset='cartpole',
    num_iterations=1000,
    num_envs=8,
    rollout_steps=128,
    context_length=20,
    quick_test=False,
    # Оптимизации
    use_amp=True,
    use_compile=True,
    use_ray=False
):
    """
    Запускает GRPO обучение
    
    Args:
        pretrained_checkpoint: путь к чекпоинту с первого этапа
        env_preset: название среды
        num_iterations: количество итераций GRPO
        num_envs: количество параллельных сред/workers
        rollout_steps: шагов в каждом rollout
        context_length: длина контекста для трансформера
        quick_test: быстрый тест режим
        use_ray: использовать Ray для параллелизации
    """
    
    # Для быстрого теста
    if quick_test:
        print("\n" + "="*80)
        print("РЕЖИМ БЫСТРОГО ТЕСТА")
        print("="*80)
        num_iterations = 10
        num_envs = 4
        rollout_steps = 32
        
    print("\n" + "="*80)
    print("GRPO ОБУЧЕНИЕ (ЭТАП 2)")
    print("="*80)
    print(f"\n📊 ПАРАМЕТРЫ:")
    print(f"  Среда: {env_preset}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Num envs/workers: {num_envs}")
    print(f"  Rollout steps: {rollout_steps}")
    print(f"  Context length: {context_length}")
    
    # Проверка Windows для torch.compile
    is_windows = sys.platform.startswith('win')
    if is_windows and use_compile:
        print(f"\n⚠️  Windows: torch.compile будет отключен")
        use_compile = False
    
    # Проверка Ray
    if use_ray and not RAY_AVAILABLE:
        print(f"\n⚠️  Ray не доступен! Установите: pip install ray")
        print(f"    Используется последовательное выполнение")
        use_ray = False
    
    print(f"\n🚀 ОПТИМИЗАЦИИ:")
    print(f"  Mixed Precision (AMP): {use_amp and torch.cuda.is_available()}")
    print(f"  torch.compile: {use_compile and hasattr(torch, 'compile')}")
    print(f"  Ray parallelization: {use_ray}")
    if is_windows:
        print(f"    ⚠️  torch.compile отключен (Windows)")
    if use_ray:
        print(f"    🚀 Ожидаемое ускорение сбора роллаутов: ~6-8x")
    
    # ========================================================================
    # ЭТАП 1: Загрузка предобученной модели
    # ========================================================================
    print("\n" + "="*80)
    print("ЭТАП 1: ЗАГРУЗКА ПРЕДОБУЧЕННОЙ МОДЕЛИ")
    print("="*80)
    
    if not os.path.exists(pretrained_checkpoint):
        print(f"\n❌ ОШИБКА: Чекпоинт не найден: {pretrained_checkpoint}")
        print("\nСначала обучите модель на supervised learning:")
        print("  python run_gym_pipeline_optimized.py --env cartpole")
        print("\nЗатем укажите путь к чекпоинту:")
        print("  python run_grpo.py --checkpoint checkpoints/.../best_model.pt")
        sys.exit(1)
    
    print(f"Загружаем чекпоинт: {pretrained_checkpoint}")
    checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')
    
    config = checkpoint['config']
    
    print(f"\n✓ Чекпоинт загружен")
    print(f"  Observation dim: {config['obs_dim']}")
    print(f"  Action dim: {config['action_dim']}")
    print(f"  Embed dim: {config['embed_dim']}")
    print(f"  Num layers: {config['num_layers']}")
    
    # Создаем policy модель
    policy = DecisionTransformer(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        num_kv_heads=config['num_kv_heads'],
        num_experts=config['num_experts'],
        max_seq_len=config.get('max_seq_len', 512),
        dropout=config.get('dropout', 0.1)
    )
    
    policy.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Веса загружены в policy")
    
    # ========================================================================
    # ЭТАП 2: Создание Value Network
    # ========================================================================
    print("\n" + "="*80)
    print("ЭТАП 2: СОЗДАНИЕ VALUE NETWORK")
    print("="*80)
    
    value_net = ValueNetwork(
        obs_dim=config['obs_dim'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['embed_dim'] * 2
    )
    
    total_params = sum(p.numel() for p in value_net.parameters())
    print(f"✓ Value Network создан: {total_params:,} параметров")
    
    # ========================================================================
    # ЭТАП 3: Конфигурация GRPO
    # ========================================================================
    print("\n" + "="*80)
    print("ЭТАП 3: КОНФИГУРАЦИЯ GRPO")
    print("="*80)
    
    grpo_config = {
        # Модель
        'obs_dim': config['obs_dim'],
        'action_dim': config['action_dim'],
        'embed_dim': config['embed_dim'],
        'num_layers': config['num_layers'],
        'num_heads': config['num_heads'],
        'num_kv_heads': config['num_kv_heads'],
        'num_experts': config['num_experts'],
        'max_seq_len': config.get('max_seq_len', 512),
        'dropout': config.get('dropout', 0.1),
        'context_length': context_length,
        
        # Среда
        'num_envs': num_envs,
        'rollout_steps': rollout_steps,
        
        # GRPO параметры
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'group_size': 8,  # Размер группы для группового нормирования
        
        # Обучение
        'num_epochs': 4,  # Эпох на каждой итерации
        'batch_size': 64,
        'policy_lr': 3e-5,  # Меньше чем на supervised learning
        'value_lr': 1e-4,
        'weight_decay': 0.01,
        
        # Оптимизации
        'use_amp': use_amp and torch.cuda.is_available(),
        'use_compile': use_compile and hasattr(torch, 'compile'),
        
        # Логирование
        'save_interval': 10,
        'checkpoint_dir': f'checkpoints/grpo_{env_preset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'log_dir': f'logs/grpo_{env_preset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        'seed': 42
    }
    
    print("GRPO конфигурация:")
    print(f"  Gamma: {grpo_config['gamma']}")
    print(f"  GAE Lambda: {grpo_config['gae_lambda']}")
    print(f"  Clip epsilon: {grpo_config['clip_epsilon']}")
    print(f"  Policy LR: {grpo_config['policy_lr']}")
    print(f"  Value LR: {grpo_config['value_lr']}")
    print(f"  Group size: {grpo_config['group_size']}")
    
    # ========================================================================
    # ЭТАП 4: Создание функции среды
    # ========================================================================
    print("\n" + "="*80)
    print("ЭТАП 4: НАСТРОЙКА СРЕДЫ")
    print("="*80)
    
    def env_fn():
        """Функция для создания среды"""
        return create_gym_environment(env_preset, seed=None)
    
    # Тестируем создание среды
    test_env = env_fn()
    print(f"✓ Среда создается корректно: {test_env.env_name}")
    test_env.close()
    
    # ========================================================================
    # ЭТАП 5: GRPO ОБУЧЕНИЕ
    # ========================================================================
    print("\n" + "="*80)
    print("ЭТАП 5: ЗАПУСК GRPO ОБУЧЕНИЯ")
    print("="*80)
    
    trainer = GRPOTrainer(
        policy_model=policy,
        value_model=value_net,
        env_fn=env_fn,
        config=grpo_config,
        use_ray=use_ray
    )
    
    print("\n🚀 Начинаем GRPO обучение...")
    print(f"{'='*80}\n")
    
    trainer.train(num_iterations=num_iterations)
    
    # ========================================================================
    # ИТОГИ
    # ========================================================================
    print("\n" + "="*80)
    print("🎉 GRPO ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*80)
    
    print(f"\n📁 РЕЗУЛЬТАТЫ:")
    print(f"  Чекпоинты: {grpo_config['checkpoint_dir']}")
    print(f"  Логи: {grpo_config['log_dir']}")
    
    print(f"\n💡 ДЛЯ ПРОСМОТРА ЛОГОВ:")
    print(f"  tensorboard --logdir {grpo_config['log_dir']}")
    
    print(f"\n📊 ДЛЯ ТЕСТИРОВАНИЯ:")
    print(f"  python test_grpo_agent.py --checkpoint {grpo_config['checkpoint_dir']}/iteration_*.pt")
    
    print(f"\n{'='*80}\n")
    
    return grpo_config['checkpoint_dir']


def main():
    parser = argparse.ArgumentParser(
        description='GRPO обучение (этап 2) - обучение с подкреплением'
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Путь к чекпоинту с первого этапа (supervised learning)')
    parser.add_argument('--env', type=str, default='cartpole',
                        choices=['cartpole', 'lunar_lander', 'mountain_car', 'acrobot', 'pendulum'],
                        help='Preset среды')
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='Количество итераций GRPO')
    parser.add_argument('--num_envs', type=int, default=8,
                        help='Количество параллельных сред/workers')
    parser.add_argument('--rollout_steps', type=int, default=128,
                        help='Шагов в каждом rollout')
    parser.add_argument('--context_length', type=int, default=20,
                        help='Длина контекста')
    
    # Оптимизации
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Использовать Mixed Precision')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Отключить Mixed Precision')
    parser.add_argument('--use_compile', action='store_true', default=True,
                        help='Использовать torch.compile')
    parser.add_argument('--no_compile', action='store_false', dest='use_compile',
                        help='Отключить torch.compile')
    
    # Ray параллелизация
    parser.add_argument('--use_ray', action='store_true', default=False,
                        help='Использовать Ray для параллельного сбора роллаутов (6-8x ускорение)')
    parser.add_argument('--no_ray', action='store_false', dest='use_ray',
                        help='Использовать последовательное выполнение')
    
    parser.add_argument('--quick_test', action='store_true',
                        help='Быстрый тест')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 ЗАПУСК GRPO ОБУЧЕНИЯ")
    print("="*80)
    print("\nПараметры:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Проверка CUDA
    if torch.cuda.is_available():
        print(f"\n✓ CUDA доступна")
        print(f"  Устройство: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n⚠️  CUDA недоступна, обучение на CPU")
    
    # Проверка Ray
    if args.use_ray:
        if not RAY_AVAILABLE:
            print(f"\n⚠️  Ray не установлен!")
            print(f"    Установите: pip install 'ray[default]'")
            print(f"    Используется последовательное выполнение")
            args.use_ray = False
        else:
            print(f"\n✓ Ray доступен")
            print(f"  Ожидаемое ускорение: ~6-8x для сбора роллаутов")
    
    try:
        checkpoint_dir = run_grpo_training(
            pretrained_checkpoint=args.checkpoint,
            env_preset=args.env,
            num_iterations=args.num_iterations,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            context_length=args.context_length,
            quick_test=args.quick_test,
            use_amp=args.use_amp,
            use_compile=args.use_compile,
            use_ray=args.use_ray
        )
        
        print("\n✓ УСПЕШНО ЗАВЕРШЕНО!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n✗ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()