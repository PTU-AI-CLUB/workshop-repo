import torch
from torch import mode, nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
from collections import defaultdict
import argparse
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
EPOCHS = 10

def train_step(model: nn.Module,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               dataloader: DataLoader) -> Tuple[float]:
    
    model.train()
    model.to(DEVICE)
    train_loss, train_acc = 0, 0
    for X_batch, y_batch in dataloader:
        
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += torch.sum(torch.argmax(y_pred, dim=-1).squeeze() == y_batch.squeeze()).item()/ BATCH_SIZE
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)


    return train_loss, train_acc

@torch.no_grad()
def eval_step(model: nn.Module,
               loss_fn: nn.Module,
               dataloader: DataLoader) -> Tuple[float]:
    
    model.eval()
    model.to(DEVICE)
    eval_loss, eval_acc = 0, 0
    
    for X_batch, y_batch in dataloader:
        
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        eval_loss += loss.item()
        eval_acc += torch.sum(torch.argmax(y_pred, dim=-1).squeeze() == y_batch.squeeze()).item()/ BATCH_SIZE
    
    eval_loss /= len(dataloader)
    eval_acc /= len(dataloader)

    return eval_loss, eval_acc

def train(model: nn.Module,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          epochs: int) -> Dict[str, List[float]]:
    
    
    results = defaultdict(list)

    for epoch in range(1, epochs+1):
        
        train_loss, train_acc = train_step(model=model,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           dataloader=train_dataloader)
        
        print(f"epoch {epoch} train_loss:{train_loss:.3f} train_acc:{train_acc}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
    
    return results


if __name__ == "__main__":
    from model import DigitClassifier
    from data import load_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", required=True,
                        metavar="", type=str)
    args = parser.parse_args()

    model = DigitClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=LEARNING_RATE
    )

    train_loader, test_loader = load_data()

    results = train(model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    train_dataloader=train_loader,
                    test_dataloader=test_loader,
                    epochs=EPOCHS)
    
    
    if "weights" not in os.listdir(".") and not os.path.isdir("./weights"):
        os.mkdir("./weights")

    file_name = args.file_name
    torch.save(
        obj=model.state_dict,
        f=f"./weights/{file_name}"
    )
    print(f"Model saved at ./weights/{file_name}")