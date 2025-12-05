import numpy as np
import torch
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def apply_moving_average(scores, window_size=5):
    """
    对 SPE 分数序列进行滑动平均平滑
    :param scores: 1D numpy array, 原始 SPE 分数
    :param window_size: 窗口大小 (建议 3~10)
    :return: 平滑后的 SPE
    """
    if len(scores) < window_size:
        return scores
    # 使用 'valid' 模式会缩短序列，'same' 模式会产生边缘效应
    # 工业上通常用简易的 valid 卷积，然后前面补 pad，或者直接用 same
    # 这里推荐 'same' 模式保持长度一致，方便对齐
    return np.convolve(scores, np.ones(window_size)/window_size, mode='same')

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
    lr = learning_rate * (0.5 ** ((epoch - 1) // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Updating learning rate to {}'.format(lr))