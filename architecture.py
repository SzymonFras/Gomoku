import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroGomokuNet(nn.Module):
    def __init__(self, board_size=15):
        super().__init__()
        self.board_size = board_size
        self.shared_core = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.GELU()
        )
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared_core(x)
        p = F.gelu(self.policy_conv(x))
        p = p.view(-1, 2 * self.board_size * self.board_size)
        policy_logits = self.policy_fc(p)
        v = F.gelu(self.value_conv(x))
        v = v.view(-1, 1 * self.board_size * self.board_size)
        v = F.gelu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, policy_logits, value_preds, target_pis, target_rewards):
        value_loss = F.mse_loss(value_preds.view(-1), target_rewards)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(target_pis * log_probs) / target_pis.size(0)
        return value_loss + policy_loss