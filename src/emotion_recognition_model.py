import torch
import torch.nn as nn

class EmotionRecognition(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 48x48 input -> conv1+pool(2) -> 24x24 -> conv2+pool(2) -> 12x12
        self.fc2 = nn.Linear(128, 7)  # 7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise
    def forward(self, x):
      x = torch.relu(self.conv1(x))
      x = torch.max_pool2d(x, 2)
      x = torch.relu(self.conv2(x))
      x = torch.max_pool2d(x, 2)

      x = x.view(x.size(0), -1)

      x = torch.relu(self.fc1(x))
      x = self.fc2(x)
      return x