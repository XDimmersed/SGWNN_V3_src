"""
SGWCN Trainer
Handles model training, validation, and checkpointing
"""

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
    """Trainer for SGWCN model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Initialize trainer
        
        Args:
            model: SGWCN model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Early stopping state
        self.early_stopping_patience = config.get('patience', 20)
        self.early_stopping_min_delta = config.get('min_delta', 1e-4)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(config['save_dir'], exist_ok=True)
        
        # Setup graceful shutdown
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
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
        """Clean up GPU memory and resources"""
        print("🧹 Cleaning up resources...")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✅ GPU memory cleared")
        
        # Delete large objects
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
        """Setup optimizer based on config"""
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
        """Setup learning rate scheduler based on config"""
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
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Check for shutdown request during training
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
        """Validate model"""
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
        """Save model checkpoint"""
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
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config['save_dir'], 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载scaler状态（如果存在且scaler已初始化）
        if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)  # 默认值0.0
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
    
    def train(self):
        """Train model with graceful shutdown support"""
        print(f"Starting training for {self.config['num_epochs']} epochs")
        print(f"Training on {self.device}")
        print("💡 Press Ctrl+C to gracefully stop training and save checkpoint")
        
        try:
            for epoch in range(self.current_epoch, self.config['num_epochs']):
                # Check for shutdown request
                if self.shutdown_requested:
                    print("\n🛑 Shutdown requested. Saving checkpoint...")
                    self.save_checkpoint()
                    print("✅ Checkpoint saved. Exiting gracefully.")
                    break
                
                self.current_epoch = epoch
                
                # Train
                train_loss, train_acc = self.train_epoch()
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                
                # Check for shutdown after training
                if self.shutdown_requested:
                    print("\n🛑 Shutdown requested after training. Saving checkpoint...")
                    self.save_checkpoint()
                    print("✅ Checkpoint saved. Exiting gracefully.")
                    break
                
                # Validate
                val_loss, val_acc = self.validate()
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                # Check for shutdown after validation
                if self.shutdown_requested:
                    print("\n🛑 Shutdown requested after validation. Saving checkpoint...")
                    self.save_checkpoint()
                    print("✅ Checkpoint saved. Exiting gracefully.")
                    break
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                # Print progress
                print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Early stopping check
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
                
                # Save checkpoint
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    print(f"New best validation accuracy: {val_acc:.2f}%")
                    # 立即保存best模型，不等待save_freq
                    self.save_checkpoint(is_best=True)
                
                # Check for early stopping
                if early_stop:
                    print("💾 Saving final checkpoint...")
                    self.save_checkpoint()
                    print("✅ Training stopped early to prevent overfitting")
                    break
                
                # Save checkpoint more frequently for better monitoring
                if (epoch + 1) % self.config.get('save_freq', 5) == 0:  # Every 5 epochs instead of 10
                    self.save_checkpoint(is_best=False)  # 保存latest，不是best
                
                # Plot training curves every epoch
                plot_freq = self.config.get('plot_freq', 1)  # Default: every epoch
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
            
            # Save final checkpoint if training completed normally
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
            # Always cleanup resources
            self.cleanup() 