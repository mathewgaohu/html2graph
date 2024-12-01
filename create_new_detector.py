"""Create a new detector that combines dom-detector and gcn-autoencoder.

"""

import os
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric.nn as gnn
from bs4 import BeautifulSoup
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.data import Data
from tqdm import tqdm

HTML2GRAPH_BASE_DIR = os.getcwd()
sys.path.insert(0, os.path.join(HTML2GRAPH_BASE_DIR, "src"))
DETECTORS_BASE_DIR = "/home/exouser/GitHub/phish_detectors"
sys.path.insert(0, os.path.join(DETECTORS_BASE_DIR, "src"))


from phish_detectors.data.dataset import HtmlDataset
from phish_detectors.data.preprocess import get_dom
from phish_detectors.models import CNNClassifier
from phish_detectors.utils.interface import AbstractDetector, PhishDetector
from phish_detectors.utils.scores import compute_classifier_scores_from_Y

from gnn_layers import GCNDecoder, GCNEncoder
from html2graph.html2graph import Html2Graph

# Get cpu or gpu device for training.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")


# Load the trained dom detector
model = CNNClassifier(
    1000,
    178,
    emb_dim=128,
    dropout_rate=0.5,
    filters=256,
    kernel_size=5,
    pool_size=5,
    strides=4,
    neurons1=128,
    neurons2=64,
)
model.load_state_dict(
    torch.load("/home/exouser/GitHub/phish_detectors/outputs/cnn_dom/model6.pth")
)
model = model.to(DEVICE)
model.eval()
detector = PhishDetector(
    preprocess=get_dom,
    transform=torch.load(
        "/home/exouser/GitHub/phish_detectors/outputs/cnn_dom/transform.pth"
    ),
    model=model,
)


# load the gae
ae = gnn.GAE(
    encoder=GCNEncoder(918, 64),
    decoder=GCNDecoder(64, 918),
)
ae.encoder.load_state_dict(
    torch.load("/home/exouser/GitHub/html2graph/outputs/GCN-GCN/encoder127.pth")
)
ae.decoder.load_state_dict(
    torch.load("/home/exouser/GitHub/html2graph/outputs/GCN-GCN/decoder127.pth")
)
ae = ae.to(DEVICE)
ae.eval()


class GAE2Detector(AbstractDetector):
    def __init__(
        self, model: gnn.GAE, recon_threshold: float, html2data: Callable[[str], Data]
    ):
        self.model = model
        self.recon_threshold = recon_threshold
        self.html2data = html2data
        self.threshold = 0.5

    def score(self, html_str: str) -> float:
        self.model.eval()
        try:
            data = self.html2data(html_str)
            with torch.no_grad():
                x = data.x.to(DEVICE)
                edge_index = data.edge_index.to(DEVICE)
                z = self.model.encode(x, edge_index)
                loss = nn.MSELoss()(x, self.model.decode(z, edge_index))
                loss = float(loss)
        except Exception as e:
            return 0.49  # pass when there is error
        score = 1.0 / (1.0 + loss / self.recon_threshold)
        return score


class DetectorAfterDetector(AbstractDetector):
    def __init__(
        self,
        detector1: AbstractDetector,
        detector2: AbstractDetector,
        benign_threshold: float = 0.0,
    ):
        self.detector1 = detector1
        self.detector2 = detector2
        self.threshold = detector1.threshold * detector2.threshold
        self.benign_threshold = benign_threshold

    def score(self, html_str: str) -> float:
        score1 = self.detector1.score(html_str)
        if score1 > self.detector1.threshold or score1 < self.benign_threshold:
            return score1
        else:
            score2 = self.detector2.score(html_str)
            return self.detector1.threshold * score2


h2g = Html2Graph()
ae_detector = GAE2Detector(
    ae,
    recon_threshold=0.005,
    html2data=lambda html_str: h2g(BeautifulSoup(html_str, "html.parser")),
)
new_detector = DetectorAfterDetector(detector, ae_detector, benign_threshold=0.0)

# Debug: test on a adv sample.
ad_df = pd.read_csv(
    "/home/exouser/GitHub/raze_to_the_ground_aisec23/outputs/attack_cnn_dom_train/annotations.csv"
)
ad_dataset = HtmlDataset(ad_df, "")
html_str, _ = ad_dataset[0]
detector.score(html_str)  # 0.45 < 0.5
new_detector.score(html_str)  # 0.35 > 0.25 # slow

# Fast evaluate the new-detector on the test set.
# Directly apply models to tensors. Advoid re-building tensors.
X = torch.load("/media/volume/sdb/detector_data/dom_X.pt").to(torch.int)
Y = torch.load("/media/volume/sdb/detector_data/dom_Y.pt").to(torch.int)
test_idx = np.load("/media/volume/sdb/detector_data/test_idx.npy")
test_dataset = TensorDataset(X[test_idx], Y[test_idx])
model.eval()
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
pred_Y = np.array([])
with torch.no_grad():
    for x, y in tqdm(test_loader, total=len(test_loader)):
        x = x.to(DEVICE)
        pred_y = model(x).to("cpu")
        pred_Y = np.r_[pred_Y, pred_y.numpy().reshape(-1)]
detector_results = pred_Y

test_data_list = torch.load("/media/volume/sdb/html2graph_data/test_graph_data.pt")
ae.eval()
loss_list = []
with torch.no_grad():
    for data in tqdm(test_data_list, total=len(test_data_list)):
        try:
            x = data.x.to(DEVICE)
            edge_index = data.edge_index.to(DEVICE)
            z = ae.encode(x, edge_index)
            loss = nn.MSELoss()(x, ae.decode(z, edge_index))
            loss_list.append(float(loss))
        except Exception as e:  # Unable to convert html to graph data or process
            loss_list.append(None)
loss_list = np.asarray(loss_list).astype(float)
ae_results = loss_list

# want the classfication scores.
recon_threshold = 0.005
ae_scores = 1.0 / (1.0 + ae_results / recon_threshold)
test_Y = Y[test_idx].numpy().astype(int)
original_decision = (detector_results > 0.5).astype(int)
new_decision = np.logical_or(detector_results > 0.5, ae_scores > 0.5).astype(int)

print("Original scores on test samples.")
acc, pre, rec, f1, auc = compute_classifier_scores_from_Y(test_Y, original_decision)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

print("New scores on test samples.")
acc, pre, rec, f1, auc = compute_classifier_scores_from_Y(test_Y, new_decision)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
np.save("old_test_pred_Y.npy", original_decision)
np.save("new_test_pred_Y.npy", new_decision)

# Get test_TP_idx (used for attack)
test_TP_idx = test_idx[np.logical_and(new_decision, test_Y)]
np.save("new_test_TP_idx.npy", test_TP_idx)
