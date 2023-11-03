from model import DigitClassifier
from data import MNISTDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import argparse

BATCH_SIZE = 64
TEST_PATH = "./data/test.csv"

def make_predictions(model_name: str,
                     pred_file_name: str):
    
    model = DigitClassifier()
    model.load_state_dict(torch.load(f=f"./weights/{model_name}"))

    test_set = MNISTDataset(path=TEST_PATH,
                           train=False)
    test_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             batch_size=BATCH_SIZE)

    model.eval()
    predictions = []
    with torch.inference_mode():
        for X_batch in test_loader:
            batch_preds = model(X_batch)
            batch_preds = torch.argmax(batch_preds, dim=-1).reshape(shape=[len(batch_preds),])
            predictions.extend(list(batch_preds.numpy()))
    
    sub_df = pd.DataFrame(
        data = [[i+1, predictions[i]] for i in range(len(predictions))],
        columns=["ImageId", "Label"]
    )

    sub_df.to_csv(f"./data/predictions/{pred_file_name}",
                  index=False)
    print("File saved successfully")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        metavar="", type=str)
    parser.add_argument("--pred_file", required=True,
                        metavar="", type=str)
    
    args = parser.parse_args()

    make_predictions(model_name=args.weights,
                     pred_file_name=args.pred_file)