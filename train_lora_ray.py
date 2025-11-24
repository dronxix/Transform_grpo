import os
import ray
import gymnasium as gym
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
import re
import shutil

# --- КОНФИГУРАЦИЯ ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./qwen_rl_checkpoints" # Папка для чекпоинтов

# Опции обучения
USE_DORA = True       # True = DoRA (лучше), False = обычная LoRA
FULL_FINETUNE = False # Если True, отключает LoRA/DoRA и учит все (нужно много VRAM!)

MAX_HISTORY = 5
GROUP_SIZE = 8        # Больше батч = стабильнее градиент
MAX_STEPS = 500       # CartPole-v1 макс 500
LEARNING_RATE = 5e-6  # Для DoRA/FullFT лучше поменьше
NUM_ITERATIONS = 50
SAVE_EVERY = 5        # Сохранять каждые N итераций

# --- 1. Ray Worker ---
@ray.remote
class EnvWorker:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        
    def reset(self):
        self.state, _ = self.env.reset()
        self.history = [] 
        self.done = False
        self.total_reward = 0
        return self._get_obs_text()

    def step(self, action):
        if self.done:
            return self._get_obs_text(), self.total_reward, True, {}
            
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.state = next_state
        self.total_reward += reward
        
        if terminated or truncated:
            self.done = True
            
        return self._get_obs_text(), self.total_reward, self.done, {}
    
    def _get_obs_text(self):
        obs = self.state
        text = f"Pos:{obs[0]:.2f}, Vel:{obs[1]:.2f}, Angle:{obs[2]:.2f}, AngVel:{obs[3]:.2f}"
        self.history.append(text)
        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)
        return self.history, self.total_reward

# --- 2. Утилиты ---
def parse_action(text):
    match = re.search(r"<action>\s*(\d+)\s*</action>", text)
    if match:
        try:
            a = int(match.group(1))
            if a in [0, 1]: return a
        except: pass
    if "1" in text[-10:]: return 1
    if "0" in text[-10:]: return 0
    return np.random.choice([0, 1])

def format_batch_prompts(batch_histories, tokenizer):
    SYSTEM_PROMPT = "You are a RL agent. Balance the pole. Output <thought>...</thought> and <action>0 or 1</action>."
    batch_texts = []
    for history in batch_histories:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        context_str = "\n".join([f"S{i}:{h}" for i, h in enumerate(history)])
        user_content = f"History:\n{context_str}\nAction?"
        messages.append({"role": "user", "content": user_content})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_texts.append(text)
    return batch_texts

# --- 3. Функции сохранения ---
def save_checkpoint(model, tokenizer, iteration, reward, is_best=False):
    """Сохраняет адаптеры (или полную модель) и токенизатор"""
    
    # Имя папки
    if is_best:
        save_path = os.path.join(OUTPUT_DIR, "best_model")
        print(f"🔥 New Best Reward ({reward:.1f})! Saving to {save_path}...")
    else:
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint_{iteration}")
        print(f"💾 Saving checkpoint to {save_path}...")
        
    # Создаем папку если нет
    os.makedirs(save_path, exist_ok=True)
    
    # Сохраняем модель (PEFT сохранит только адаптеры, FullFT - всё)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Сохраняем метаданные (какая награда была)
    with open(os.path.join(save_path, "metrics.txt"), "w") as f:
        f.write(f"iteration: {iteration}\nreward: {reward}\n")

# --- 4. Main Setup ---
# Отключаем логи ray
ray.init(ignore_reinit_error=True, log_to_driver=False)

print(f"Загрузка Qwen (DoRA={USE_DORA}, FullFT={FULL_FINETUNE})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Загружаем базу
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto",
    dtype=torch.float16,
    # use_cache=False нужно для Gradient Checkpointing (экономия памяти), если FullFT
    use_cache=not FULL_FINETUNE 
)

if not FULL_FINETUNE:
    # Настройка DoRA / LoRA
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Учим все линейные слои
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM",
        use_dora=USE_DORA # <-- Вот тут включается магия DoRA
    )
    model = get_peft_model(model, peft_config)
    print("PEFT (LoRA/DoRA) mode active.")
    model.print_trainable_parameters()
else:
    print("Full Fine-Tuning mode active. Warning: High VRAM usage.")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
workers = [EnvWorker.remote() for _ in range(GROUP_SIZE)]

# --- 5. Training Loop ---

best_avg_reward = -float('inf')

print(f"\n🚀 Start Training. Output dir: {OUTPUT_DIR}")

try:
    for it in range(1, NUM_ITERATIONS + 1):
        print(f"\n--- Iteration {it}/{NUM_ITERATIONS} ---")
        
        # --- A. Rollout (Сбор данных) ---
        # Сброс
        obs_data = ray.get([w.reset.remote() for w in workers])
        histories = [d[0] for d in obs_data]
        trajectories = [[] for _ in range(GROUP_SIZE)]
        active_indices = list(range(GROUP_SIZE))
        finished_rewards = [0] * GROUP_SIZE
        
        step_count = 0
        while active_indices:
            step_count += 1
            # Batch Inference
            active_histories = [histories[i] for i in active_indices]
            prompts = format_batch_prompts(active_histories, tokenizer)
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.8, pad_token_id=tokenizer.pad_token_id)
            
            gen_ids = outputs[:, inputs.input_ids.shape[1]:]
            gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            actions = [parse_action(t) for t in gen_texts]
            
            # Env Step
            futures = [workers[active_indices[i]].step.remote(actions[i]) for i in range(len(active_indices))]
            results = ray.get(futures)
            
            next_active = []
            for i, (res_obs, res_reward, res_done, _) in enumerate(results):
                agent_idx = active_indices[i]
                
                # Сохраняем тензоры на CPU для экономии VRAM
                trajectories[agent_idx].append({
                    "input_ids": inputs.input_ids[i].cpu(),
                    "gen_ids": gen_ids[i].cpu(),
                })
                histories[agent_idx] = res_obs[0]
                
                if res_done:
                    finished_rewards[agent_idx] = res_reward
                else:
                    if step_count < MAX_STEPS:
                        next_active.append(agent_idx)
                    else:
                        finished_rewards[agent_idx] = res_reward # Force stop
            active_indices = next_active

        # --- B. GRPO Update (Обучение) ---
        rewards_t = torch.tensor(finished_rewards, dtype=torch.float32, device=model.device)
        mean_r = rewards_t.mean().item()
        std_r = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean_r) / std_r
        
        print(f"  Rewards: {finished_rewards} | Mean: {mean_r:.1f}")
        
        optimizer.zero_grad()
        total_loss = 0
        
        # Обучаем батчами по траекториям (можно распараллелить, но так проще по памяти)
        for i, traj in enumerate(trajectories):
            adv = advantages[i]
            if abs(adv.item()) < 0.1: continue # Пропускаем "средние" результаты, учимся только на хороших/плохих
            
            for step_data in traj:
                inp = step_data["input_ids"].to(model.device).unsqueeze(0)
                gen = step_data["gen_ids"].to(model.device).unsqueeze(0)
                full = torch.cat([inp, gen], dim=1)
                
                out = model(full)
                logits = out.logits[:, inp.shape[1]-1 : full.shape[1]-1, :]
                
                loss = -F.cross_entropy(logits.transpose(1, 2), gen, reduction='none').sum() * adv
                loss = loss / (len(traj) * GROUP_SIZE)
                loss.backward()
                total_loss += loss.item()
                
        optimizer.step()
        print(f"  Loss: {total_loss:.4f}")
        
        # --- C. Checkpointing (Сохранение) ---
        
        # 1. Сохраняем лучшую модель
        if mean_r > best_avg_reward:
            best_avg_reward = mean_r
            save_checkpoint(model, tokenizer, it, mean_r, is_best=True)
            
        # 2. Сохраняем регулярно
        if it % SAVE_EVERY == 0:
            save_checkpoint(model, tokenizer, it, mean_r, is_best=False)

except KeyboardInterrupt:
    print("Остановка пользователем. Сохраняем текущее состояние...")
    save_checkpoint(model, tokenizer, it, mean_r, is_best=False)

ray.shutdown()
