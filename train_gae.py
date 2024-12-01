"""Train GNN AutoEncoder.

Requirement: 
- Graph data. 
- training/valication indices.

Run this script:
nohup python train_gae.py > train_gae.out 2>train_gae.log & disown

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric.nn as gnn
from torch import nn
from torch_geometric.data import Data
from tqdm.auto import tqdm

from gnn_layers import (
    GATDecoder,
    GATEncoder,
    GCNDecoder,
    GCNEncoder,
    VariationalGCNEncoder,
    VGAEDecoder,
)

##################################################
########## Parameters (can change)      ##########
##################################################
MODEL = "GCN-GCN"
OUTPUT_DIR = f"outputs/{MODEL}"
##################################################

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")


# Create data
all_data = torch.load("/media/volume/sdb/html2graph_data/ad_graph_data.pt")
train_idx = np.load("/media/volume/sdb/html2graph_data/ad_train_idx.npy")
val_idx = np.load("/media/volume/sdb/html2graph_data/ad_val_idx.npy")
train_data = [all_data[i] for i in train_idx]
val_data = [all_data[i] for i in val_idx]

for data in train_data:
    print("example data = ", data)
    break


# Create AE Model
in_channels = data.x.shape[1]
out_channels = 64
if MODEL == "GCN":
    model = gnn.GAE(encoder=GCNEncoder(in_channels, out_channels)).to(DEVICE)
elif MODEL == "GCN-GCN":
    model = gnn.GAE(
        encoder=GCNEncoder(in_channels, out_channels),
        decoder=GCNDecoder(out_channels, in_channels),
    ).to(DEVICE)
print(model)


# Define training process
epochs = 200
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(data_list: list[Data]) -> float:
    model.train()
    sum_loss = 0.0
    for data in tqdm(data_list):
        x = data.x.to(DEVICE)
        edge_index = data.edge_index.to(DEVICE)

        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = nn.MSELoss()(x, model.decode(z, edge_index))
        # loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()
        sum_loss += float(loss)
    return sum_loss / len(data_list)


def test(data_list: list[Data]) -> float:
    model.eval()
    sum_loss = 0.0
    with torch.no_grad():
        for data in data_list:
            x = data.x.to(DEVICE)
            edge_index = data.edge_index.to(DEVICE)
            z = model.encode(x, edge_index)
            loss = nn.MSELoss()(x, model.decode(z, edge_index))
            # loss = model.recon_loss(z, edge_index)
            sum_loss += float(loss)
    return sum_loss / len(data_list)


# Main: Training
if __name__ == "__main__":
    torch.manual_seed(42)
    train_loss_list = []
    val_loss_list = []
    for epoch in range(1, epochs + 1):
        train_loss = train(train_data)
        val_loss = test(val_data)
        print(
            f"Epoch {epoch}: Train Loss = {train_loss:>10.6f}, Val Loss = {val_loss:>10.6f}"
        )
        torch.save(
            model.encoder.state_dict(), os.path.join(OUTPUT_DIR, f"encoder{epoch}.pth")
        )
        torch.save(
            model.decoder.state_dict(), os.path.join(OUTPUT_DIR, f"decoder{epoch}.pth")
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    # Plot loss
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), train_loss_list, label="training")
    plt.plot(np.arange(1, epochs + 1), val_loss_list, label="validatoin")
    plt.xlabel("epochs")
    plt.ylabel("MSE")
    plt.title("Graph Auto Encoder Training Process")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "train_loss.png"))
    np.save(
        os.path.join(OUTPUT_DIR, "train_loss.npy"),
        np.asarray([train_loss_list, val_loss_list]),
    )
