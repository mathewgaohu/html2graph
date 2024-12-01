from train_gae import *
import seaborn as sns


##################################################
########## Parameters (can change)      ##########
##################################################
MODEL_IDX = 127  # 37  # minimum val loss
##################################################


# Load trained model
model.encoder.load_state_dict(
    torch.load(os.path.join(OUTPUT_DIR, f"encoder{MODEL_IDX}.pth"))
)
model.decoder.load_state_dict(
    torch.load(os.path.join(OUTPUT_DIR, f"decoder{MODEL_IDX}.pth"))
)
# model.encoder = torch.load(os.path.join(OUTPUT_DIR, f"encoder{MODEL_IDX}.pth"))
# model.decoder = torch.load(os.path.join(OUTPUT_DIR, f"decoder{MODEL_IDX}.pth"))


# Create test data
ad_data = all_data
test_data = torch.load("/media/volume/sdb/html2graph_data/test_graph_data.pt")
test_Y = torch.load("/media/volume/sdb/detector_data/dom_Y.pt")[
    np.load("/media/volume/sdb/detector_data/test_idx.npy")
]

for data in test_data:
    print("data = ", data)
    break


# Define test process
def test_every(data_list: list[Data]) -> np.ndarray[float]:
    model.eval()
    loss_list = []
    with torch.no_grad():
        for data in tqdm(data_list):
            try:
                x = data.x.to(DEVICE)
                edge_index = data.edge_index.to(DEVICE)
                z = model.encode(x, edge_index)
                loss = nn.MSELoss()(x, model.decode(z, edge_index))
                # loss = model.recon_loss(z, edge_index)
                # loss = (x - model.decode(z, edge_index)).norm() / x.norm()
                loss_list.append(float(loss))
            except Exception as e:  # Unable to convert html to graph data or process
                loss_list.append(None)
    loss_list = np.asarray(loss_list).astype(float)
    return loss_list


adv_loss_list = test_every(ad_data)
test_loss_list = test_every(test_data)

adv_val_loss_list = adv_loss_list[val_idx]
phish_loss_list = test_loss_list[test_Y == 1]
benign_loss_list = test_loss_list[test_Y == 0]


hist_options = {
    "binrange": (0.0, 0.03),
    "stat": "percent",
    "alpha": 0.6,
    "linewidth" : 0,
}
plt.figure()
sns.histplot(phish_loss_list, color="r", label="Phishing", **hist_options)
sns.histplot(benign_loss_list, color="g", label="Legitimate", **hist_options)
sns.histplot(adv_val_loss_list, color="y", label="Adversarial", **hist_options)
plt.xlabel("Reconstruction Error")
plt.ylabel("Sample Percentage")
plt.title("GCAE Performance on Different HTMLs")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, f"test_loss{MODEL_IDX}.png"))


# debug
np.count_nonzero(np.isnan(phish_loss_list))
np.count_nonzero(np.isnan(benign_loss_list))
