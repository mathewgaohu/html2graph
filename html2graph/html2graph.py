import copy
import os
import re

import numpy as np
import torch
from bs4 import BeautifulSoup, Comment
from bs4.element import Tag
from torch_geometric.data import Data
from transformers import BertModel, BertTokenizer

# Load stored top tags and top attributes. 
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "top_tags.txt"), "r") as infile:
    top_tags = infile.read().split("\n")
with open(os.path.join(current_dir, "top_attrs.txt"), "r") as infile:
    top_attrs = infile.read().split("\n")


def get_text(tag: Tag) -> str:
    """Get text from a tag. Exclude the texts that belong to its children tag."""
    all_text = tag.find_all(string=True, recursive=False)
    concat_text = " ".join([s.strip() for s in all_text])
    return re.sub(r"\s+", " ", concat_text.strip())


class Html2Graph:

    def __init__(
        self,
        text_max_len: int = 512,  # must <=512 (limitation from BERT)
        neglect_style_text: bool = True,
        remove_comments: bool = True,
    ):
        self.text_max_len = text_max_len
        self.neglect_style_text = neglect_style_text
        self.remove_comments = remove_comments

        self.tag_list = top_tags
        self.attrs_list = top_attrs

        self.n_tags = len(self.tag_list) + 1  # index -1 is for out-of-vocab
        self.n_attrs = len(self.attrs_list)

        # Initialize BERT tokenizer and model
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        self.model: BertModel = BertModel.from_pretrained("bert-base-uncased")

        # store output of blank text for faster behavior.
        inputs = self.tokenizer.encode_plus(
            "",
            add_special_tokens=True,
            truncation=True,
            max_length=self.text_max_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            self.blank_text_emb: np.ndarray = self.model(**inputs)[0][0][0].numpy()

    def __call__(self, soup: BeautifulSoup) -> Data:
        """
        Transform a beautifulsoup object to a graph.

        Each node represents a tag. Each node owns a feature vector: [tag_vector, attrs_vector, text_vector]
        - tag_vector (len=self.n_tags): One-hot encoding of the tag name.
        - attrs_vector (len=self.n_attrs): Represents whether an attribute exists. Ommits the values of attributes.
        - text_vector (len=768): Text embedding via BERT.

        Args:
            soup: Parsed HTML object via beautifulsoup.

        Returns:
            torch_geometric.data.Data: Graph data.

        """

        soup = copy.deepcopy(soup)

        if self.remove_comments:
            for comments in soup.findAll(text=lambda text: isinstance(text, Comment)):
                comments.extract()

        # Initialize lists to store node features and edge indices
        node_features = []
        edge_index = []

        # Function to recursively traverse the DOM tree and create node features
        def traverse_dom_tree(element: Tag):
            my_id = len(node_features)

            # One hot encoding for tags.
            tag_encoding = np.zeros(self.n_tags)
            if element.name in self.tag_list:
                tag_encoding[self.tag_list.index(element.name)] = 1.0
            else:
                tag_encoding[-1] = 1.0

            # Get attributes of the current DOM node and generate one-hot encoding
            attrs = element.attrs
            attrs_encoding = np.asarray([(x in attrs) for x in self.attrs_list])

            # Get text of the current DOM node and embed using BERT
            text = get_text(element)
            if self.neglect_style_text and element.name == "style":
                text = ""

            if text == "":
                text_embedding = self.blank_text_emb
            else:
                inputs = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.text_max_len,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    text_embedding = self.model(**inputs)[0][0][0].numpy()

            # Concatenate one-hot encoding and BERT embedding as node features
            node_features.append(
                np.concatenate([tag_encoding, attrs_encoding, text_embedding])
            )

            # Traverse child nodes
            for i, child in enumerate(element.children):
                if child.name is not None:
                    child_id = len(node_features)  # this will be the id of the child.
                    edge_index.append([my_id, child_id])
                    traverse_dom_tree(child)

        # Get the root element of the DOM tree
        root = soup.find("html")
        # TODO: Deal with cases with no html tag. We should still get a graph.

        # Start traversing the DOM tree
        traverse_dom_tree(root)

        # Convert node features and edge indices to torch tensors
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        edge_index = (
            torch.tensor(np.array(edge_index), dtype=torch.long).t().contiguous()
        )

        # Create the graph using torch_geometric
        data = Data(x=x, edge_index=edge_index)
        return data
