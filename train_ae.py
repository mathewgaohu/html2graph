import time
from tqdm.auto import tqdm
import pandas as pd
import torch
from torch import nn
from torch.utils.data import random_split

from dev.html2graph import Html2Graph
from env_config import *
from phish_detector.data import HTMLDataset
import torch_geometric.nn as gnn
from dev.gnn_layers import GCNEncoder, GCNDecoder, VariationalGCNEncoder, VGAEDecoder, GATEncoder, GATDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

data_dir = os.path.join(ROOT_DIR, "rl_data_train")
df = pd.DataFrame(data=[[name, 1] for name in os.listdir(data_dir)],
                  columns=["file", "label"])
ds = HTMLDataset(df, data_dir, transform=Html2Graph())

train_size = 1000
ds_train, _ = random_split(ds, [train_size, len(ds) - train_size],
                           generator=torch.Generator().manual_seed(42))

example_data, _ = ds_train[0]

num_features = example_data.x.shape[1]
out_channels = 64

# model = gnn.GAE(encoder=GCNEncoder(num_features, out_channels)).to(device)
model = gnn.GAE(encoder=GCNEncoder(num_features, out_channels),
                decoder=GCNDecoder(out_channels, num_features)).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def train():
    model.train()  # sets the module into training mode
    sum_loss = 0.
    for idx, (data, _) in tqdm(enumerate(ds_train), total=len(ds_train)):
        X = data.x.to(device)
        edge_index = data.edge_index.to(device)

        optimizer.zero_grad()  # resets the gradients of all optimized torch.Tensors
        z = model.encode(X, edge_index)  # runs the encoder and computes node-wise latent variables (GCNEncoder)
        X_decode = model.decode(z, edge_index)
        loss = nn.MSELoss()(X, X_decode)
        # loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()
    sum_loss += float(loss)
    return sum_loss


epochs = 20

for epoch in range(1, epochs+1):
    loss = train()
    print(f"Epoch {epoch}: Loss = {loss:>10.2f}")
    torch.save(model.encoder, f"encoder_{epoch}.pth")
    torch.save(model.decoder, f"decoder_{epoch}.pth")
