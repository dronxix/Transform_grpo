import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import gymnasium as gym
import re

# ПУТЬ К СОХРАНЕННОЙ МОДЕЛИ
CHECKPOINT_PATH = "./qwen_rl_checkpoints/best_model" 
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Загрузка из {CHECKPOINT_PATH}...")

# 1. Загружаем токенизатор из чекпоинта
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)

# 2. Загружаем базовую модель
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    dtype=torch.float16
)

# 3. Накладываем обученные слои (LoRA/DoRA)
# Если использовали Full Finetune, то PeftModel не нужен, просто грузим model из пути
try:
    model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
    print("Адаптеры LoRA/DoRA успешно загружены.")
except:
    print("Не удалось загрузить адаптеры, возможно это Full Finetune или ошибка пути.")

# --- Играем ---
env = gym.make("CartPole-v1", render_mode="human") # human покажет окно с игрой
state, _ = env.reset()
history = []

def get_action(s, h):
    text = f"Pos:{s[0]:.2f}, Angle:{s[2]:.2f}"
    h.append(text)
    if len(h)>5: h.pop(0)
    
    msgs = [{"role": "system", "content": "You are a RL agent. Balance the pole. <thought>...</thought> <action>..."}]
    hist_str = "\n".join(h)
    msgs.append({"role": "user", "content": f"History:\n{hist_str}\nAction?"})
    
    inputs = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, temperature=0.1) # Низкая темп для теста
    
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"Model thought: {txt.split('Action?')[-1]}") # Печатаем мысль
    
    # Парсинг
    if "<action>0</action>" in txt: return 0
    if "<action>1</action>" in txt: return 1
    return 0 if "0" in txt[-10:] else 1

done = False
while not done:
    action = get_action(state, history)
    state, reward, term, trunc, _ = env.step(action)
    done = term or trunc

print("Игра окончена!")
env.close()
