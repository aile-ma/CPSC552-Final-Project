import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import openslide
from openslide import OpenSlideError

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

import matplotlib.pyplot as plt

RNASEQ_PATH = "/gpfs/gibbs/project/cpsc452/cpsc452_jl4286/filtered_rnaseq_data.npy"
RNA_IDS     = "/gpfs/gibbs/project/cpsc452/cpsc452_jl4286/filtered_rna_ids.npy"
CLIN_CSV    = "/gpfs/gibbs/project/cpsc452/cpsc452_jl4286/cleaned_clinical.csv"

WSI_DIRS = [
    "/gpfs/gibbs/project/cpsc452/cpsc452_jl4286/image",
    "/gpfs/gibbs/project/cpsc452/cpsc452_am4289/image",
    "/gpfs/gibbs/project/cpsc452/cpsc452_yg427/data/image_svs_manual",
    "/gpfs/gibbs/project/cpsc452/cpsc452_yy743/image_svs_manual",
]

def get_case_id(filename):
    return "-".join(os.path.basename(filename).split("-")[:3])

wsi_map = defaultdict(list)
for d in WSI_DIRS:
    for fp in glob.glob(os.path.join(d, "*.svs")):
        case_id = get_case_id(fp)
        wsi_map[case_id].append(fp)


excluded = [
    "/gpfs/gibbs/project/cpsc452/cpsc452_am4289/image/TCGA-14-0781-01B-01-BS1.svs",
    "/gpfs/gibbs/project/cpsc452/cpsc452_am4289/image/TCGA-12-3652-01A-01-TS1.svs",
    "/gpfs/gibbs/project/cpsc452/cpsc452_am4289/image/TCGA-14-0789-01Z-00-DX6.svs",
    "/gpfs/gibbs/project/cpsc452/cpsc452_am4289/image/TCGA-12-3653-01Z-00-DX3.svs",
    "/gpfs/gibbs/project/cpsc452/cpsc452_am4289/image/TCGA-14-0786-01B-01-TS1.svs",
    "/gpfs/gibbs/project/cpsc452/cpsc452_am4289/image/TCGA-12-5299-01A-01-BS1.svs",
    "/gpfs/gibbs/project/cpsc452/cpsc452_yg427/data/image_svs_manual/TCGA-14-1795-01Z-00-DX4.svs",
    "/gpfs/gibbs/project/cpsc452/cpsc452_yg427/data/image_svs_manual/TCGA-26-5133-01Z-00-DX1.svs",
    "/gpfs/gibbs/project/cpsc452/cpsc452_yg427/data/image_svs_manual/TCGA-02-0003-01Z-00-DX3.svs"
]


valid_wsi_map = {}
for pid, paths in wsi_map.items():
    kept = [p for p in paths if p not in excluded]
    if kept:
        valid_wsi_map[pid] = kept

clin = pd.read_csv(CLIN_CSV)
CLIN = ["days_to_birth", "gender", "race_asian", "race_black", "race_unknown", "race_white"]
clin = clin.dropna(subset=CLIN + ['event'])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
clin[CLIN] = scaler.fit_transform(clin[CLIN])
clin.head()

rna_mat = np.load("/gpfs/gibbs/project/cpsc452/cpsc452_jl4286/filtered_rnaseq_data.npy")
rna_ids = np.load("/gpfs/gibbs/project/cpsc452/cpsc452_jl4286/filtered_rna_ids.npy", allow_pickle=True)
pid2idx = {pid: i for i, pid in enumerate(rna_ids)}
print("RNA matrix:", rna_mat.shape)

patch_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_patches(svs_path, transform=None, size=224, max_patches=5):
    try:
        slide = openslide.OpenSlide(svs_path)
        for level in sorted(range(len(slide.level_dimensions)), reverse=True):
            W, H = slide.level_dimensions[level]
            if W >= size and H >= size:
                break
        else:
            return
        count = 0
        for y in range(0, H, size):
            for x in range(0, W, size):
                if count >= max_patches:
                    return
                img = slide.read_region((x, y), level, (size, size)).convert("RGB")
                if np.array(img).mean() > 245:
                    continue
                patch = transform(img) if transform else patch_tf(img)
                if patch.shape != (3, 224, 224):
                    continue
                yield patch
                count += 1
    except:
        return

class GBMDataset(Dataset):
    def __init__(self, clin_df, rna_mat, pid2idx, wsi_map, clin_vars):
        self.clin_df = clin_df.reset_index(drop=True)
        self.rna_mat = rna_mat
        self.pid2idx = pid2idx
        self.wsi_map = wsi_map
        self.clin_vars = clin_vars

    def __len__(self): return len(self.clin_df)

    def __getitem__(self, idx):
        row = self.clin_df.iloc[idx]
        pid = row['submitter_id']
        clin_tensor = torch.tensor(row[self.clin_vars].values.astype(np.float32))
        rna_tensor = torch.tensor(self.rna_mat[self.pid2idx[pid]], dtype=torch.float32)
        wsi_paths = self.wsi_map.get(pid, [])
        time = float(row['survival_time'])
        event = int(row['event'])
        return pid, clin_tensor, rna_tensor, wsi_paths, time, event

def collate_fn(batch):
    _, clin_vs, rna_vs, wsi_lists, times, events = zip(*batch)
    Xc = torch.stack(clin_vs)
    Xr = torch.stack(rna_vs)
    wf = []
    for wsi_list in wsi_lists:
        feats = []
        for svs in wsi_list:
            try:
                for p in extract_patches(svs, transform=patch_tf):
                    with torch.no_grad():
                        f = wsi_encoder(p.unsqueeze(0).to(device))
                    if f.dim() == 2 and f.shape[1] == wsi_encoder.out_dim:
                        feats.append(f.cpu())
            except: continue
        wf.append(torch.mean(torch.stack(feats), dim=0) if feats else torch.zeros(wsi_encoder.out_dim))
    try:
        Xw = torch.stack(wf)
    except:
        Xw = torch.zeros(len(wf), wsi_encoder.out_dim)
    return Xc, Xr, Xw, torch.tensor(times), torch.tensor(events)

class RNAEncoder(nn.Module):
    def __init__(self, G, hidden=256, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(G, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, out), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class WSLEncoder(nn.Module):
    def __init__(self, out=128):
        super().__init__()
        res = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.back = nn.Sequential(*list(res.children())[:-1])
        self.proj = nn.Linear(res.fc.in_features, out)
        self.out_dim = out
    def forward(self, x):
        f = self.back(x).view(x.size(0), -1)
        return self.proj(f)

class ZorroAttentionFusion(nn.Module):
    def __init__(self, cD, rD=128, wD=128, h=256):
        super().__init__()
        D = 128
        self.c_proj = nn.Linear(cD, D)
        self.r_proj = nn.Identity()
        self.w_proj = nn.Identity()
        self.attn = nn.Sequential(
            nn.Linear(3 * D, 3),
            nn.Softmax(dim=1)
        )
        self.fuse = nn.Sequential(
            nn.Linear(D, h), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(h, 1)
        )
    def forward(self, c, r, w):
        c = F.normalize(self.c_proj(c), dim=1)
        r = F.normalize(self.r_proj(r), dim=1)
        w = F.normalize(self.w_proj(w), dim=1)
        modals = torch.stack([c, r, w], dim=1)
        concat = torch.cat([c, r, w], dim=1)
        alpha = self.attn(concat).unsqueeze(-1)
        fused = (modals * alpha).sum(dim=1)
        return self.fuse(fused).squeeze(-1)

def cox_loss(risk, T, E):
    order = torch.argsort(T, descending=True)
    r = risk[order]
    e = E[order]
    logcum = torch.logcumsumexp(r, dim=0)
    return -torch.sum(r * e - logcum * e)

# Training
clin = pd.read_csv(CLIN_CSV)
CLIN_FEATS = ["days_to_birth", "gender", "race_asian", "race_black", "race_unknown", "race_white"]
clin = clin.dropna(subset=CLIN_FEATS + ['survival_time', 'event'])
scaler = StandardScaler(); clin[CLIN_FEATS] = scaler.fit_transform(clin[CLIN_FEATS])
rna_mat = np.load(RNASEQ_PATH)
rna_ids = np.load(RNA_IDS, allow_pickle=True)
pid2idx = {pid: i for i, pid in enumerate(rna_ids)}

all_pids = [pid for pid in clin['submitter_id'] if pid in pid2idx and pid in valid_wsi_map]
train_pids, test_pids = train_test_split(all_pids, test_size=0.2, random_state=42)
train_clin = clin[clin['submitter_id'].isin(train_pids)]
test_clin  = clin[clin['submitter_id'].isin(test_pids)]

train_ds = GBMDataset(train_clin, rna_mat, pid2idx, valid_wsi_map, CLIN_FEATS)
test_ds  = GBMDataset(test_clin,  rna_mat, pid2idx, valid_wsi_map, CLIN_FEATS)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wsi_encoder = WSLEncoder(out=128).to(device)
rna_encoder = RNAEncoder(G=rna_mat.shape[1]).to(device)
fusion      = ZorroAttentionFusion(cD=len(CLIN_FEATS)).to(device)
opt = optim.Adam(list(rna_encoder.parameters()) + list(wsi_encoder.parameters()) + list(fusion.parameters()), lr=1e-4)

num_epochs = 50
val_cidx_history = []
for epoch in range(num_epochs):
    total = 0
    rna_encoder.train(); wsi_encoder.train(); fusion.train()
    for Xc, Xr, Xw, Ts, Es in train_loader:
        Xc, Xr, Xw, Ts, Es = [t.to(device) for t in (Xc, Xr, Xw, Ts, Es)]
        opt.zero_grad()
        fR = rna_encoder(Xr)
        fW = Xw.squeeze(1) if Xw.dim() == 3 else Xw
        risk = fusion(Xc, fR, fW)
        loss = cox_loss(risk, Ts, Es)
        loss.backward(); opt.step()
        total += loss.item()
    print(f"[Epoch {epoch+1}] Train Loss: {total/len(train_loader):.4f}")

    fusion.eval(); rna_encoder.eval(); wsi_encoder.eval()
    risks, Ts_all, Es_all = [], [], []
    with torch.no_grad():
        for Xc, Xr, Xw, Ts, Es in test_loader:
            Xc, Xr, Xw = [t.to(device) for t in (Xc, Xr, Xw)]
            fR = rna_encoder(Xr)
            fW = Xw.squeeze(1) if Xw.dim() == 3 else Xw
            risk = fusion(Xc, fR, fW).cpu().numpy()
            risks.extend(risk)
            Ts_all.extend(Ts.numpy())
            Es_all.extend(Es.numpy())

    cidx = concordance_index(Ts_all, -np.array(risks), Es_all)
    print(f"[Epoch {epoch+1}] Test C-Index: {cidx:.4f}")

    risk_group = np.array(risks) >= np.median(risks)
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6, 4))
    for g in [0, 1]:
        ix = risk_group == g
        label = 'High Risk' if g else 'Low Risk'
        kmf.fit(np.array(Ts_all)[ix], np.array(Es_all)[ix], label=label)
        kmf.plot(ci_show=False)
    plt.title("Kaplan-Meier by Predicted Risk")
    plt.xlabel("Days to Death"); plt.ylabel("Survival Probability")
    plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6, 5))
    import seaborn as sns
    sns.scatterplot(x=risks, y=Ts_all, hue=Es_all, palette={1: "red", 0: "blue"}, s=60)
    plt.xlabel("Predicted Risk Score")
    plt.ylabel("Actual Survival Time (Days)")
    plt.title("Predicted Risk vs. Actual Survival Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


epochs = np.arange(1, num_epochs+1)
c_index = np.array(val_cidx_history)
plt.figure(figsize=(8,4))
plt.plot(epochs, c_index, marker='o', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Validation C-index')
plt.xticks(np.arange(0, num_epochs+1, 5))
plt.ylim(c_index.min()-0.02, c_index.max()+0.02)
plt.grid(True)
plt.tight_layout()
plt.savefig("validation_cindex_curve.png", dpi=300, bbox_inches="tight")
plt.show()
