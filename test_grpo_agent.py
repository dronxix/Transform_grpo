#!/usr/bin/env python3
"""
Тестирование обученного GRPO агента
"""

import torch
import numpy as np
import argparse
from collections import deque

from model import DecisionTransformer
from gym_environment import create_gym_environment


class GRPOAgent:
    """
    Агент для inference с GRPO обученной моделью
    """
    
    def __init__(self, model, context_length=20, device='cuda'):
        self.model = model.eval()
        self.device = device
        self.context_length = context_length
        
        self.obs_history = deque(maxlen=context_length)
        self.action_history = deque(maxlen=context_length-1)
        
    def reset(self):
        """Сбрасывает историю"""
        self.obs_history.clear()
        self.action_history.clear()
    
    @torch.no_grad()
    def select_action(self, observation, temperature=1.0, deterministic=False):
        """
        Выбирает действие для данного наблюдения
        
        Args:
            observation: np.array [obs_dim]
            temperature: температура для sampling
            deterministic: если True, выбирает argmax
            
        Returns:
            action: int
        """
        # Добавляем в историю
        self.obs_history.append(observation)
        
        # Подготавливаем последовательность
        obs_seq = torch.FloatTensor(list(self.obs_history)).unsqueeze(0).to(self.device)  # [1, hist_len, obs_dim]
        
        if len(self.action_history) > 0:
            actions_seq = torch.LongTensor(list(self.action_history))
            # Добавляем START token
            start_token = torch.LongTensor([self.model.action_dim])
            actions_seq = torch.cat([start_token, actions_seq]).unsqueeze(0).to(self.device)  # [1, hist_len]
        else:
            # Только START token
            actions_seq = torch.full((1, len(self.obs_history)), self.model.action_dim, 
                                    dtype=torch.long, device=self.device)
        
        # Forward pass
        action_logits = self.model(obs_seq, actions_seq)
        last_logits = action_logits[0, -1, :]  # [action_dim]
        
        if deterministic:
            action = last_logits.argmax().item()
        else:
            probs = torch.softmax(last_logits / temperature, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()
        
        # Добавляем в историю
        self.action_history.append(action)
        
        return action


def test_agent(
    checkpoint_path,
    env_preset='cartpole',
    num_episodes=10,
    render=False,
    temperature=1.0,
    deterministic=False
):
    """
    Тестирует агента
    """
    print("="*80)
    print("ТЕСТИРОВАНИЕ GRPO АГЕНТА")
    print("="*80)
    
    # Загружаем чекпоинт
    print(f"\nЗагружаем чекпоинт: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint['config']
    
    print(f"Конфигурация:")
    print(f"  Observation dim: {config['obs_dim']}")
    print(f"  Action dim: {config['action_dim']}")
    print(f"  Context length: {config['context_length']}")
    
    # Создаем модель
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = DecisionTransformer(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        num_kv_heads=config['num_kv_heads'],
        num_experts=config['num_experts']
    ).to(device)
    
    model.load_state_dict(checkpoint['policy_state_dict'])
    print(f"✓ Модель загружена на {device}")
    
    # Создаем агента
    agent = GRPOAgent(
        model,
        context_length=config['context_length'],
        device=device
    )
    
    # Создаем среду
    env = create_gym_environment(env_preset, seed=100)
    print(f"\n✓ Среда создана: {env.env_name}")
    
    # Тестируем
    print(f"\nЗапускаем {num_episodes} эпизодов...")
    print(f"  Temperature: {temperature}")
    print(f"  Deterministic: {deterministic}")
    print(f"  Render: {render}")
    print("="*80)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        agent.reset()
        
        episode_reward = 0.0
        steps = 0
        
        for step in range(1000):
            action = agent.select_action(obs, temperature=temperature, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if render:
                # Здесь можно добавить рендеринг
                pass
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        print(f"Эпизод {episode + 1:3d}: {steps:4d} шагов, reward={episode_reward:8.2f}")
    
    env.close()
    
    # Статистика
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ")
    print("="*80)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print(f"\nНаграда:")
    print(f"  Средняя: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Мин: {min(episode_rewards):.2f}")
    print(f"  Макс: {max(episode_rewards):.2f}")
    
    print(f"\nДлина эпизода:")
    print(f"  Средняя: {mean_length:.1f} ± {std_length:.1f}")
    print(f"  Мин: {min(episode_lengths)}")
    print(f"  Макс: {max(episode_lengths)}")
    
    print(f"\n{'='*80}")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'std_length': std_length
    }


def main():
    parser = argparse.ArgumentParser(description='Тестирование GRPO агента')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Путь к чекпоинту GRPO')
    parser.add_argument('--env', type=str, default='cartpole',
                        choices=['cartpole', 'lunar_lander', 'mountain_car', 'acrobot', 'pendulum'],
                        help='Preset среды')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Количество тестовых эпизодов')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Температура для sampling')
    parser.add_argument('--deterministic', action='store_true',
                        help='Использовать детерминированную политику (argmax)')
    parser.add_argument('--render', action='store_true',
                        help='Рендерить среду')
    
    args = parser.parse_args()
    
    results = test_agent(
        checkpoint_path=args.checkpoint,
        env_preset=args.env,
        num_episodes=args.num_episodes,
        render=args.render,
        temperature=args.temperature,
        deterministic=args.deterministic
    )
    
    print("\n✓ Тестирование завершено!")


if __name__ == '__main__':
    main()
