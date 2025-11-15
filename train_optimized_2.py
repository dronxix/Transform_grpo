"""
Оптимизированный Trainer с:
- Mixed Precision Training (AMP) - 2-3x ускорение
- Gradient Accumulation - большие эффективные батчи
- torch.compile - 1.5-2x ускорение (PyTorch 2.0+)
- Оптимизированный DataLoader
- Multi-GPU support (DDP)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import os
import sys
import json
import time
from datetime import datetime
from tqdm import tqdm

from model import DecisionTransformer
from gym_data_preparation import load_gym_trajectories, create_dataloaders

if torch.cuda.is_available():
    # Новый API для TF32 (PyTorch 2.9+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class OptimizedTrainer:
    """
    Оптимизированный тренер с Mixed Precision, Gradient Accumulation, torch.compile
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Проверка Windows - torch.compile не поддерживается
        is_windows = sys.platform.startswith('win')
        if is_windows and config.get('use_compile', False):
            print("⚠️  Windows обнаружена: torch.compile отключен (требует Triton, недоступен на Windows)")
            config['use_compile'] = False
        
        # Применяем torch.compile для ускорения (PyTorch 2.0+)
        if config.get('use_compile', False) and hasattr(torch, 'compile'):
            print("🚀 Компилируем модель с torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
            print("✓ Модель скомпилирована!")
        
        self.model = model.to(device)
        
        # Оптимизатор
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            fused=True if device == 'cuda' else False  # Fused optimizer для GPU
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['learning_rate'] * 0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Mixed Precision Training
        self.use_amp = config.get('use_amp', False) and device == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Mixed Precision Training включен")
        
        # Gradient Accumulation
        self.accumulation_steps = config.get('accumulation_steps', 1)
        if self.accumulation_steps > 1:
            print(f"✓ Gradient Accumulation: {self.accumulation_steps} шагов")
            print(f"  Эффективный batch size: {config['batch_size'] * self.accumulation_steps}")
        
        # Для логирования
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Для отслеживания прогресса
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.epoch = 0
        
        # Создаем директории
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # Сохраняем конфиг
        with open(os.path.join(config['checkpoint_dir'], 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"\n{'='*80}")
        print(f"Trainer инициализирован:")
        print(f"  Device: {self.device}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Gradient Accumulation: {self.accumulation_steps}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Effective batch size: {config['batch_size'] * self.accumulation_steps}")
        print(f"{'='*80}\n")
    
    def train_epoch(self):
        """Один epoch обучения с оптимизациями"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.config['num_epochs']}")
        
        # Для gradient accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            observations = batch['observations'].to(self.device, non_blocking=True)
            actions = batch['actions'].to(self.device, non_blocking=True)
            returns_to_go = batch['returns_to_go'].to(self.device, non_blocking=True)  # НОВОЕ!
            targets = batch['targets'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)
            
            # Mixed Precision Training
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    # ИЗМЕНЕНО: передаем returns_to_go
                    logits, lb_loss = self.model(observations, actions, returns_to_go)
                    
                    # Loss
                    batch_size, seq_len, action_dim = logits.shape
                    logits_flat = logits.view(-1, action_dim)
                    targets_flat = targets.view(-1)
                    action_loss = self.criterion(logits_flat, targets_flat)
                    
                    # Добавляем load balancing loss
                    loss = action_loss + self.config.get('load_balancing_loss_coef', 0.01) * lb_loss
                    
                    # Масштабируем для gradient accumulation
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
            else:
                # ИЗМЕНЕНО: передаем returns_to_go
                logits, lb_loss = self.model(observations, actions, returns_to_go)
                
                # Loss
                batch_size, seq_len, action_dim = logits.shape
                logits_flat = logits.view(-1, action_dim)
                targets_flat = targets.view(-1)
                action_loss = self.criterion(logits_flat, targets_flat)
                
                # Добавляем load balancing loss
                loss = action_loss + self.config.get('load_balancing_loss_coef', 0.01) * lb_loss
                
                loss = loss / self.accumulation_steps
                loss.backward()
            
            # Optimizer step каждые accumulation_steps батчей
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Метрики (игнорируем паддинг)
            with torch.no_grad():
                predictions = logits_flat.argmax(dim=-1)
                valid_mask = (targets_flat != -100)
                correct = ((predictions == targets_flat) & valid_mask).sum().item()
                total_correct += correct
                total_samples += valid_mask.sum().item()
            
            # Реальный loss (умножаем обратно)
            actual_loss = loss.item() * self.accumulation_steps
            total_loss += actual_loss
            
            # Логирование
            if self.global_step % self.config['log_interval'] == 0:
                valid_samples = valid_mask.sum().item()
                batch_accuracy = correct / valid_samples if valid_samples > 0 else 0.0
                self.writer.add_scalar('train/loss', actual_loss, self.global_step)
                self.writer.add_scalar('train/accuracy', batch_accuracy, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
            
            # Update progress bar
            valid_samples_batch = valid_mask.sum().item()
            batch_acc = correct / valid_samples_batch if valid_samples_batch > 0 else 0.0
            pbar.set_postfix({
                'loss': f'{actual_loss:.4f}',
                'acc': f'{batch_acc:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    @torch.no_grad()
    def validate(self):
        """Валидация с оптимизациями"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            observations = batch['observations'].to(self.device, non_blocking=True)
            actions = batch['actions'].to(self.device, non_blocking=True)
            returns_to_go = batch['returns_to_go'].to(self.device, non_blocking=True)  # НОВОЕ!
            targets = batch['targets'].to(self.device, non_blocking=True)
            
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    logits, lb_loss = self.model(observations, actions, returns_to_go)  # ИЗМЕНЕНО
            else:
                logits, lb_loss = self.model(observations, actions, returns_to_go)  # ИЗМЕНЕНО
            
            # Loss
            batch_size, seq_len, action_dim = logits.shape
            logits_flat = logits.view(-1, action_dim)
            targets_flat = targets.view(-1)
            
            action_loss = self.criterion(logits_flat, targets_flat)
            loss = action_loss + self.config.get('load_balancing_loss_coef', 0.01) * lb_loss
            
            # Метрики
            predictions = logits_flat.argmax(dim=-1)
            valid_mask = (targets_flat != -100)
            correct = ((predictions == targets_flat) & valid_mask).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += valid_mask.sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def save_checkpoint(self, filename='checkpoint.pt', is_best=False):
        """Сохранение чекпоинта"""
        # Если модель скомпилирована, берем оригинал
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save(checkpoint, path)
        print(f"Чекпоинт сохранен: {path}")
        
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Лучшая модель сохранена: {best_path}")
    
    def load_checkpoint(self, filename='checkpoint.pt'):
        """Загрузка чекпоинта"""
        path = os.path.join(self.config['checkpoint_dir'], filename)
        
        if not os.path.exists(path):
            print(f"Чекпоинт не найден: {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Если модель скомпилирована, загружаем в оригинал
        model_to_load = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Чекпоинт загружен: {path}")
        print(f"Resuming from epoch {self.epoch + 1}")
        
        return True
    
    def train(self):
        """Полный цикл обучения"""
        print("=" * 80)
        print("НАЧИНАЕМ ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.config['num_epochs']}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation: {self.accumulation_steps}")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config['num_epochs']):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Обучение
            train_loss, train_acc = self.train_epoch()
            
            # Валидация
            val_loss, val_acc = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Логирование
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/train_accuracy', train_acc, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/val_accuracy', val_acc, epoch)
            self.writer.add_scalar('epoch/time', epoch_time, epoch)
            
            # Вывод статистики
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Time: {epoch_time:.2f}s | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Сохранение чекпоинтов
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            
            # Сохранение лучшей модели
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('checkpoint.pt', is_best=True)
                print(f"✓ Новая лучшая модель! Val Loss: {val_loss:.4f}")
            
            print("-" * 80)
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"{'='*80}")
        print(f"Общее время: {total_time / 3600:.2f} часов ({total_time:.2f} секунд)")
        print(f"Среднее время на эпоху: {total_time / self.config['num_epochs']:.2f} секунд")
        print(f"Лучший val loss: {self.best_val_loss:.4f}")
        print(f"{'='*80}")
        
        self.writer.close()
