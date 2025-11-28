"""SGWCN 训练器：管理训练、验证、保存与早停的完整流程。"""

import os
import time
import signal
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm

from ..utils.visualization import plot_training_curves


class SGWCNTrainer:
    """SGWCN 模型的训练封装类。"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        """初始化训练器并准备优化器、调度器、损失等组件。

        参数：
            model: SGWCN 模型实例。
            train_loader: 训练集 DataLoader。
            val_loader: 验证集 DataLoader。
            config: 训练配置字典（学习率/保存目录等）。
            device: 训练设备字符串。
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)
        
        # 将模型迁移到目标设备（原英文注释“Move model to device”）
        self.model = self.model.to(self.device)

        # 初始化优化器（原英文注释“Setup optimizer”）
        self.optimizer = self._setup_optimizer()

        # 初始化学习率调度器（原英文注释“Setup scheduler”）
        self.scheduler = self._setup_scheduler()

        # 配置损失函数（原英文注释“Setup loss function”）
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get('label_smoothing', 0.0)
        )

        # 配置混合精度训练（原英文注释“Setup mixed precision”）
        self.scaler = GradScaler() if config.get('use_amp', True) else None

        # 训练状态缓存（原英文注释“Training state”）
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # 早停相关状态（原英文注释“Early stopping state”）
        self.early_stopping_patience = config.get('patience', 20)
        self.early_stopping_min_delta = config.get('min_delta', 1e-4)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')

        # 创建模型保存目录（原英文注释“Create checkpoint directory”）
        os.makedirs(config['save_dir'], exist_ok=True)

        # 注册信号处理，支持平滑退出（原英文注释“Setup graceful shutdown”）
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理中断信号，触发安全退出与检查点保存。"""
        print(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
        print("💾 Saving checkpoint before exit...")
        self.shutdown_requested = True
        
        # 如果多次收到信号，强制退出
        if hasattr(self, '_signal_count'):
            self._signal_count += 1
            if self._signal_count >= 3:
                print("🛑 Multiple signals received. Force exiting...")
                sys.exit(1)
        else:
            self._signal_count = 1
    
    def cleanup(self):
        """清理 GPU 缓存与对象，防止资源泄漏。"""
        print("🧹 Cleaning up resources...")
        
        # 清理 GPU 缓存（原英文注释“Clear GPU cache”）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✅ GPU memory cleared")

        # 删除占用显存/内存的大对象（原英文注释“Delete large objects”）
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        if hasattr(self, 'criterion'):
            del self.criterion
        if hasattr(self, 'scaler') and self.scaler:
            del self.scaler
        if hasattr(self, 'scheduler') and self.scheduler:
            del self.scheduler
        if hasattr(self, 'train_loader'):
            del self.train_loader
        if hasattr(self, 'val_loader'):
            del self.val_loader
        
        print("✅ Resources cleaned up")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """根据配置选择并初始化优化器。"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """配置学习率调度策略。"""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs']
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 50),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """完成单个 epoch 的训练，返回平均损失与精度。"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            # 训练过程中检查是否收到退出信号（原英文注释翻译）
            if self.shutdown_requested:
                print(f"\n🛑 Shutdown requested during training. Stopping at batch {batch_idx}...")
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                if self.config.get('gradient_clip', 0.0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                if self.config.get('gradient_clip', 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        pbar.close()  # 确保进度条被正确关闭
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """在验证集上评估模型，返回平均损失与精度。"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc='Validation')
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Check for shutdown request during validation
            if self.shutdown_requested:
                print(f"\n🛑 Shutdown requested during validation. Stopping at batch {batch_idx}...")
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            if self.scaler is not None:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        pbar.close()  # 确保进度条被正确关闭
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, is_best: bool = False):
        """保存最新/最佳模型检查点。"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,  # 保存scaler状态
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'config': self.config
        }
        
        # 保存最新检查点（原英文注释“Save latest checkpoint”）
        latest_path = os.path.join(self.config['save_dir'], 'latest.pth')
        torch.save(checkpoint, latest_path)

        # 若指标最佳则额外保存 best 检查点（原英文注释“Save best checkpoint”）
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载已有检查点以恢复训练。"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载 scaler 状态（如果存在且 scaler 已初始化）
        if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)  # 默认值 0.0
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
    
    def train(self):
        """运行完整训练流程，支持早停与安全退出。"""
        print(f"Starting training for {self.config['num_epochs']} epochs")
        print(f"Training on {self.device}")
        print("💡 Press Ctrl+C to gracefully stop training and save checkpoint")
        
        try:
            for epoch in range(self.current_epoch, self.config['num_epochs']):
                # 每个 epoch 开始时检查退出标记
                if self.shutdown_requested:
                    print("\n🛑 Shutdown requested. Saving checkpoint...")
                    self.save_checkpoint()
                    print("✅ Checkpoint saved. Exiting gracefully.")
                    break

                self.current_epoch = epoch

                # 训练阶段
                train_loss, train_acc = self.train_epoch()
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)

                # 训练完成后再次检查退出标记
                if self.shutdown_requested:
                    print("\n🛑 Shutdown requested after training. Saving checkpoint...")
                    self.save_checkpoint()
                    print("✅ Checkpoint saved. Exiting gracefully.")
                    break

                # 验证阶段
                val_loss, val_acc = self.validate()
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

                # 验证后再次检查退出标记
                if self.shutdown_requested:
                    print("\n🛑 Shutdown requested after validation. Saving checkpoint...")
                    self.save_checkpoint()
                    print("✅ Checkpoint saved. Exiting gracefully.")
                    break

                # 更新学习率调度
                if self.scheduler:
                    self.scheduler.step()

                # 打印本轮训练/验证结果
                print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

                # 早停条件检查
                early_stop = False
                if self.config.get('early_stopping', True):
                    if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                        self.best_val_loss = val_loss
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter >= self.early_stopping_patience:
                            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
                            print(f"Validation loss hasn't improved for {self.early_stopping_patience} epochs")
                            early_stop = True
                
                # 依据验证精度保存 best/最新检查点
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    print(f"New best validation accuracy: {val_acc:.2f}%")
                    # 立即保存 best 模型，不等待 save_freq
                    self.save_checkpoint(is_best=True)

                # 若触发早停则保存并退出
                if early_stop:
                    print("💾 Saving final checkpoint...")
                    self.save_checkpoint()
                    print("✅ Training stopped early to prevent overfitting")
                    break

                # 按频率保存最新模型，便于中途观察（默认每 5 个 epoch）
                if (epoch + 1) % self.config.get('save_freq', 5) == 0:
                    self.save_checkpoint(is_best=False)  # 保存 latest 而非 best

                # 按频率绘制训练曲线（默认每个 epoch）
                plot_freq = self.config.get('plot_freq', 1)
                if (epoch + 1) % plot_freq == 0:
                    plot_training_curves(
                        self.train_losses,
                        self.val_losses,
                        self.train_accs,
                        self.val_accs,
                        save_path=os.path.join(self.config['save_dir'], 'training_curves.png')
                    )
                    if plot_freq == 1:
                        print(f"📊 训练曲线已更新: {os.path.join(self.config['save_dir'], 'training_curves.png')}")
            
            # 若训练正常结束则保存最终检查点
            if not self.shutdown_requested:
                self.save_checkpoint()
                print("\n🎉 Training completed!")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received. Saving checkpoint...")
            self.save_checkpoint()
            print("✅ Checkpoint saved. Exiting gracefully.")
        
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            print("💾 Attempting to save checkpoint...")
            try:
                self.save_checkpoint()
                print("✅ Checkpoint saved despite error.")
            except:
                print("❌ Failed to save checkpoint.")
            raise
        
        finally:
            # 无论如何都清理资源
            self.cleanup()