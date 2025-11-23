import os
import re
import torch
import gymnasium as gym
import numpy as np
import collections
from typing import Optional, List, Dict, Any

from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================

# --- НАСТРОЙКИ ЗАГРУЗКИ МОДЕЛИ ---
USE_LOCAL = False  
# True  = Использовать локальную папку (интернет не нужен).
# False = Качать с HuggingFace или брать из кэша HF.

# Путь к локальной папке (если USE_LOCAL = True)
LOCAL_MODEL_PATH = "C:/Models/Qwen2.5-1.5B-Instruct" 

# ID модели на HuggingFace (если USE_LOCAL = False)
HF_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# --- НАСТРОЙКИ СРЕДЫ И ИСТОРИИ ---
ENV_ID = "CartPole-v1"
MAX_HISTORY_STEPS = 6  # Сколько последних шагов (Obs+Action) помнить. Старое забывается.

# --- НАСТРОЙКИ ОБУЧЕНИЯ (под RTX 4090) ---
OUTPUT_DIR = "./qwen_rl_history_output"
MAX_STEPS = 300       # Количество шагов обучения
BATCH_SIZE = 4        # Размер батча на устройство
GRAD_ACCUM = 4        # Накопление градиентов (эффективный батч = 4 * 4 = 16)
NUM_GENERATIONS = 8   # GRPO генерирует 8 вариантов ответа на 1 промпт

# ==========================================
# 2. ПОДГОТОВКА И УТИЛИТЫ
# ==========================================

def get_model_source():
    """Определяет, откуда брать модель."""
    if USE_LOCAL:
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(
                f"❌ ОШИБКА: Режим USE_LOCAL=True, но путь не найден: {LOCAL_MODEL_PATH}"
            )
        print(f"✅ [OFFLINE MODE] Используем локальную модель: {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH, {"local_files_only": True}
    else:
        print(f"🌐 [ONLINE/CACHE MODE] Используем HF ID: {HF_MODEL_ID}")
        return HF_MODEL_ID, {"local_files_only": False}

def format_history_prompt(history: list) -> list:
    """
    Преобразует историю (список словарей role/content) в формат сообщений для ChatML.
    """
    # Системный промпт: объясняем задачу и формат вывода XML
    messages = [
        {"role": "system", "content": (
            "You are a reinforcement learning agent controlling a CartPole system. "
            "Your goal is to balance the pole. "
            "Analyze the history of observations. "
            "Output ONLY the next action as an integer (0 or 1) inside <action> tags, like <action>1</action>."
        )}
    ]
    # Добавляем историю взаимодействия (Obs -> Action -> Obs ...)
    messages.extend(history)
    return messages

# ==========================================
# 3. ГЕНЕРАЦИЯ ДАТАСЕТА С ИСТОРИЕЙ
# ==========================================

def build_dataset_with_history(tokenizer, num_samples=200):
    """
    Создает датасет, симулируя короткие эпизоды игры, чтобы наполнить контекст.
    """
    env = gym.make(ENV_ID)
    dataset_data = []
    
    print(f"🔄 Generating {num_samples} samples with history context...")
    
    for _ in range(num_samples):
        # Очередь с автоматическим удалением старых элементов (реализация "забывания")
        # maxlen * 2, так как храним пары User(Obs) и Assistant(Action)
        history_buffer = collections.deque(maxlen=MAX_HISTORY_STEPS * 2)
        
        obs, _ = env.reset()
        current_obs_str = f"Observation: {np.array2string(obs, precision=3)}"
        
        # Случайная длина "разогрева" от 0 до 5 шагов
        warmup_steps = np.random.randint(0, 6)
        
        # --- ФАЗА РАЗОГРЕВА (Наполняем историю) ---
        for _ in range(warmup_steps):
            # 1. Записываем наблюдение
            history_buffer.append({"role": "user", "content": current_obs_str})
            
            # 2. Выбираем случайное действие (имитация прошлого опыта)
            action = env.action_space.sample()
            action_str = f"<action>{action}</action>"
            history_buffer.append({"role": "assistant", "content": action_str})
            
            # 3. Шаг среды
            obs, _, terminated, truncated, _ = env.step(action)
            current_obs_str = f"Observation: {np.array2string(obs, precision=3)}"
            
            if terminated or truncated:
                obs, _ = env.reset()
                history_buffer.clear()
                current_obs_str = f"Observation: {np.array2string(obs, precision=3)}"
        
        # --- ФАЗА ФОРМИРОВАНИЯ ПРОМПТА ДЛЯ ОБУЧЕНИЯ ---
        # Добавляем последнее наблюдение, на которое модель должна ответить
        history_buffer.append({"role": "user", "content": current_obs_str})
        
        # Превращаем в текст
        messages = format_history_prompt(list(history_buffer))
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        dataset_data.append({
            "prompt": prompt_text,
            # Сохраняем "сырое" состояние среды, чтобы reward_function могла его восстановить
            "raw_state": env.unwrapped.state 
        })

    env.close()
    return dataset_data

# ==========================================
# 4. REWARD FUNCTION (ЛОГИКА НАГРАДЫ)
# ==========================================

def reward_function(prompts, completions, **kwargs):
    rewards = []
    env = gym.make(ENV_ID)
    
    for prompt, completion in zip(prompts, completions):
        # 1. Парсинг действия
        # Ищем <action>X</action> или просто число
        action_match = re.search(r"<action>(\d+)</action>", completion)
        if not action_match:
            action_match = re.search(r"(\d+)", completion)
            
        valid_format = False
        action = 0
        
        if action_match:
            try:
                action = int(action_match.group(1))
                if action in [0, 1]:
                    valid_format = True
            except:
                pass
        
        if not valid_format:
            rewards.append(-1.0) # Штраф за мусор на выходе
            continue

        # 2. Восстановление состояния среды (Observation Reconstruction)
        # GRPO передает текст, но не объект среды.
        # Нам нужно вытащить ПОСЛЕДНЕЕ наблюдение из текста промпта.
        # Формат в тексте: "Observation: [ 0.01  -0.02 ... ]"
        try:
            # Берем последний "Observation:", чтобы игнорировать старую историю
            last_obs_text = prompt.split("Observation:")[-1]
            obs_match = re.search(r"\[([\d\.\s\-\w]+)\]", last_obs_text)
            
            if obs_match:
                # Парсим числа обратно в массив
                obs_values = np.fromstring(obs_match.group(1), sep=' ')
                
                # ХАК для CartPole: Принудительно ставим состояние
                env.reset()
                env.unwrapped.state = obs_values 
                
                # 3. Выполняем действие
                _, r, terminated, _, _ = env.step(action)
                
                # Расчет награды
                current_reward = float(r)
                
                # Бонус за правильный формат XML (помогает модели быстрее понять структуру)
                if "<action>" in completion:
                    current_reward += 0.5
                
                # Сильный штраф за падение
                if terminated:
                    current_reward = -5.0
                
                rewards.append(current_reward)
            else:
                # Если не смогли найти наблюдение в промпте (странная ошибка)
                rewards.append(0.0)
                
        except Exception as e:
            # print(f"Env Error: {e}") # Можно раскомментировать для отладки
            rewards.append(-0.5)
            
    env.close()
    return rewards

# ==========================================
# 5. ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
# ==========================================

def main():
    # Настройка Ray: убираем лишний шум в логах
    os.environ["RAY_DEDUP_LOGS"] = "0"
    
    # 1. Получаем путь к модели
    model_path, model_kwargs = get_model_source()
    
    # 2. Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_path, **model_kwargs)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Готовим данные
    # Используем datasets из HuggingFace для совместимости с TRL
    raw_data = build_dataset_with_history(tokenizer, num_samples=500)
    from datasets import Dataset
    dataset = Dataset.from_list(raw_data)

    # 4. Конфигурация DoRA (Weight-Decomposed Low-Rank Adaptation)
    # Значительно эффективнее обычной LoRA для обучения новым задачам
    peft_config = LoraConfig(
        r=32,               # Ранг (Rank)
        lora_alpha=64,      # Alpha (обычно rank * 2)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True       # Включаем DoRA
    )

    # 5. Конфигурация GRPO Trainer
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,          # Аккуратный Learning Rate
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        fp16=True,                   # FP16 для ускорения на RTX 4090
        logging_steps=10,
        save_steps=100,
        # GRPO параметры
        num_generations=NUM_GENERATIONS, # Размер группы для сэмплирования
        max_completion_length=32,        # Нам нужно только короткое действие
        beta=0.04,                       # KL penalty (чтобы не уходила далеко от базовой модели)
        # vLLM и Ray параметры
        use_vllm=False,                   # Использовать быстрый движок генерации
        # vllm_gpu_memory_utilization=0.3, # 30% VRAM под Ray/vLLM, 70% под Trainer
    )

    print(f"\n🚀 Starting GRPO Training on {ENV_ID}...")
    print(f"   Model: {model_path}")
    print(f"   History Context: {MAX_HISTORY_STEPS} steps")
    print(f"   Device: RTX 4090 (Allocating ~30% for Inference, ~70% for Train)\n")

    # 6. Инициализация тренера
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # 7. Запуск
    trainer.train()
    
    # 8. Сохранение
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_path)
    print(f"\n✅ Training finished! Model saved to: {final_path}")

if __name__ == "__main__":
    main()
