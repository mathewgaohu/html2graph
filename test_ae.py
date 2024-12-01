"""Test ae on a set of new htmls. Look at the relative change in x (fetures). """
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import random_split

from dev.html2graph import Html2Graph
from env_config import *
from phish_detector.data import HTMLDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

data_dir = os.path.join(ROOT_DIR, "rl_data_train")
df = pd.DataFrame(data=[[name, 1] for name in os.listdir(data_dir)],
                  columns=["file", "label"])
ds = HTMLDataset(df, data_dir, transform=Html2Graph())

train_size = 1000
test_size = 10
ds_train, ds_test, _ = random_split(ds, [train_size, test_size, len(ds) - train_size - test_size],
                                    generator=torch.Generator().manual_seed(42))

errs = []
rel_errs = []
for i in range(1, 20 + 1):
    print(f"Open encoder from epoch {i}")
    encoder = torch.load(f"encoder_{i}.pth").to(device)
    decoder = torch.load(f"decoder_{i}.pth").to(device)

    encoder.eval()
    decoder.eval()

    err = 0.
    rel_err = 0.
    for idx, (data, _) in enumerate(ds_test):
        print("Apply to first test sample.")
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        with torch.no_grad():
            z = encoder(x, edge_index)  # runs the encoder and computes node-wise latent variables (GCNEncoder)
            x_decode = decoder(z, edge_index)
            loss = nn.MSELoss()(x, x_decode)
            rel_loss = loss / (x.norm() / np.sqrt(x.numel()))

        print("x :", x.numpy()[0, -5:])
        print("x_decode :", x_decode.numpy()[0, -5:])
        print("loss :", loss.item())
        print("loss :", rel_loss.item())

        err += loss.item()
        rel_err += rel_loss.item()

    errs.append(err)
    rel_errs.append(rel_err)

print(f"Summary - Test on {test_size} samples.")
for i in range(1, 20 + 1):
    print(f"Epoch {i:>4d}: Loss = {errs[i-1]:>10.6f}, Relative Error = {rel_errs[-1]/test_size:>10.6f}")


