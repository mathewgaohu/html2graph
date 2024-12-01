"""Convert adversarial examples to graph data.

Run this script:
nohup python convert_html_to_graph_adv.py > convert_html_to_graph.out 2>&1 & disown
"""

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm

from html2graph.html2graph import Html2Graph

PHISH_DETECTORS_BASE_DIR = "/home/exouser/GitHub/phish_detectors"
sys.path.insert(0, os.path.join(PHISH_DETECTORS_BASE_DIR, "src"))
from phish_detectors.data.dataset import HtmlDataset

##################################################
########## Parameters (can change)      ##########
##################################################
# Options
CREATE_ANNOTATIONS = False
# Directories
DATA_DIR = (
    "/home/exouser/GitHub/raze_to_the_ground_aisec23/outputs/attack_cnn_dom_train"
)
HTML_DATA_DIR = (
    "/home/exouser/GitHub/raze_to_the_ground_aisec23/outputs/attack_cnn_dom_train/adv"
)
OUTPUT_DIR = "/media/volume/sdb/html2graph_data"
# Parameters
##################################################
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create html info
if CREATE_ANNOTATIONS:
    ids = np.asarray(os.listdir(HTML_DATA_DIR)).astype(int)
    df = pd.DataFrame(
        {
            "id": ids,
            "html_path": [
                os.path.join(HTML_DATA_DIR, str(i), "index.html") for i in ids
            ],
            "url_path": [os.path.join(HTML_DATA_DIR, str(i), "url.txt") for i in ids],
            "label": [1] * len(ids),
        }
    )
    df.to_csv(os.path.join(DATA_DIR, "annotations.csv"))
else:
    df = pd.read_csv(os.path.join(DATA_DIR, "annotations.csv"))

# Create HTML dataset
dataset = HtmlDataset(df, DATA_DIR)

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


# Accelerate code with multi-processing - Failed - Areadly using all CPUs.
# def process_data(idx):
#     try:
#         html_str, _ = dataset[idx]
#         soup = BeautifulSoup(html_str, "html.parser")
#         data = h2g(soup)
#         return idx, data
#     except Exception as e:
#         print(f"Error ({idx}): ", e)
#         return idx, None
# data_list_unordered = []
# with Pool() as pool:
#     # Assuming html_dataset is a list of (x, y) tuples
#     with tqdm(total=len(dataset)) as pbar:
#         # Define a callback function to update the progress bar
#         def update():
#             pbar.update()
#         # Apply sync map with callback
#         results = pool.imap_unordered(process_data, range(len(dataset)))
#         for result in results:
#             i, data = result
#             data_list_unordered.append((i, data))
#             update()
# index_ordered = np.argsort([x[0] for x in data_list_unordered])
# data_list = [data_list_unordered[i][1] for i in index_ordered]


torch.save(data_list, os.path.join(OUTPUT_DIR, "ad_graph_data.pt"))

# Load data
# data_list = torch.load(os.path.join(OUTPUT_DIR, "ad_graph_data.pt"))
# data_list
