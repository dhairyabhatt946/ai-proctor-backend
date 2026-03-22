import torch.nn as nn
import torch.nn.functional as F

class ProctoringCNN(nn.Module):
    def __init__(self):
        super(ProctoringCNN, self).__init__()

        # The Eyes (Feature Extraction)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # The Compressor
        self.pool = nn.MaxPool2d(2, 2)

        # The Brain (Linear Layers)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 16) # Outputs the 16 exact coordinates

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 64 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Raw coordinate output

        return x
