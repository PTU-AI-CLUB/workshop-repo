import torch
from torch import mode, nn

class DigitClassifier(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=784, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10)
        )
    
    def forward(self,
                X: torch.Tensor) -> float:
        return self.net(X)
    
if __name__ == "__main__":
    model = DigitClassifier()
    print(model)