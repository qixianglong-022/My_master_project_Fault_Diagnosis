import numpy as np
import torch

class EarlyStopping:
    """
    早停机制：当验证集损失在 patience 个 epoch 内不再下降时，停止训练。
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def adjust_learning_rate(optimizer, epoch, learning_rate):
    # 简单的衰减策略：每过一个 epoch 衰减一点点，或者根据 epoch 阶段衰减
    # 这里采用 type1: 半衰减
    lr = learning_rate * (0.5 ** ((epoch - 1) // 2)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Updating learning rate to {}'.format(lr))