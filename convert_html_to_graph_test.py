"""Convert test examples to graph data.

Run this script:
nohup python convert_html_to_graph_test.py > convert_html_to_graph.out 2>&1 & disown
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm

from html2graph.html2graph import Html2Graph
from torch.utils.data.dataset import Subset

PHISH_DETECTORS_BASE_DIR = "/home/exouser/GitHub/phish_detectors"
sys.path.insert(0, os.path.join(PHISH_DETECTORS_BASE_DIR, "src"))
from phish_detectors.data.dataset import HtmlDataset

##################################################
########## Parameters (can change)      ##########
##################################################
# Options
# Directories
DATA_DIR = "/media/volume/sdb/detector_data"
OUTPUT_DIR = "/media/volume/sdb/html2graph_data"
# Parameters
##################################################
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create html info
df = pd.read_csv(os.path.join(DATA_DIR, "annotations.csv"))

# Create HTML dataset
full_dataset = HtmlDataset(df, DATA_DIR)
test_index = np.load(os.path.join(DATA_DIR, "test_idx.npy"))
dataset = Subset(full_dataset, test_index)

# Convert to graphs
h2g = Html2Graph()
data_list = []
for idx in tqdm(range(len(dataset))):
    try:
        html_str, _ = dataset[idx]
        soup = BeautifulSoup(html_str, "html.parser")
        data = h2g(soup)
        data_list.append(data)
    except Exception as e:
        print(f"Error ({idx}): ", e)
        data_list.append(None)


torch.save(data_list, os.path.join(OUTPUT_DIR, "test_graph_data.pt"))

# Load data
# data_list = torch.load(os.path.join(OUTPUT_DIR, "ad_graph_data.pt"))
# data_list
