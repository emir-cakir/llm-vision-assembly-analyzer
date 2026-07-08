"""
STAGE 3b (v2 - temporal model) - Colab: extract_sequences.py'nin uretttigi
video-basi sirali feature dizileri uzerine bir BiLSTM egitir. Sabit-pencere
bagimsiz siniflandirma yerine, model her zaman adiminda ONCESINI VE
SONRASINI (iki yonlu baglam) gorerek tahmin yapar - bu, pencere uzunlugunun
aksiyon suresine tam uymadigi durumlarda (kisa aksiyonlar) cok daha
dayanikli sonuc verir.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report

# --------------------------------------------------------------------------
# Drive mount
# --------------------------------------------------------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive/ikea_project")
SEQ_DIR = DRIVE_ROOT / "sequences"

EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-3
HIDDEN = 256
DROPOUT = 0.3
IGNORE_INDEX = -100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Kullanilan cihaz:", DEVICE)

# --------------------------------------------------------------------------
# Sekans dosyalarini yukle, train/test olarak ayir
# --------------------------------------------------------------------------
import json
with open(SEQ_DIR / "label_vocab.json", "r", encoding="utf-8") as f:
    LABEL2IDX_ORIG = json.load(f)
IDX2LABEL_ORIG = {v: k for k, v in LABEL2IDX_ORIG.items()}

# --------------------------------------------------------------------------
# Ikinci seviye birlestirme: asiri nadir 2 sinifi (18 ve 148 zaman-adimi)
# semantik olarak en yakin olduklari sinifla birlestiriyoruz. Bu kadar az
# ornekle LSTM (agirliklandirma ile bile) saglikli bir oruntu ogrenemez.
# --------------------------------------------------------------------------
FURTHER_MERGE = {
    "NA": "NA",
    "other": "other",
    "align part": "align part",
    "attach part": "attach part",
    "flip part": "flip part",
    "pick up part": "handle part",
    "lay down part": "handle part",
    "rotate table": "move table",
    "push table": "move table",
    "slide part": "slide part",
    "spin leg": "spin leg",
    "tighten leg": "tighten leg",
}
LABEL2IDX = {label: i for i, label in enumerate(sorted(set(FURTHER_MERGE.values())))}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}
num_classes = len(LABEL2IDX)

remap_table = torch.zeros(len(LABEL2IDX_ORIG), dtype=torch.long)
for orig_label, orig_idx in LABEL2IDX_ORIG.items():
    remap_table[orig_idx] = LABEL2IDX[FURTHER_MERGE[orig_label]]

print(f"Etiketler tekrar birlestirildi: {len(LABEL2IDX_ORIG)} -> {num_classes} sinif")

seq_files = [fp for fp in sorted(SEQ_DIR.glob("*.pt")) if fp.name != "temporal_model.pt"]
print(f"{len(seq_files)} sekans dosyasi bulundu.")

train_items, test_items = [], []
for fp in seq_files:
    data = torch.load(fp)
    if data["features"].shape[0] == 0:
        continue
    data["labels"] = remap_table[data["labels"]]   # eski (12) -> yeni (10) etiket index'i
    if data["subset"] == "training":
        train_items.append(data)
    else:
        test_items.append(data)

print(f"Train video sayisi: {len(train_items)}, Test video sayisi: {len(test_items)}")
print(f"Train toplam zaman adimi: {sum(d['features'].shape[0] for d in train_items)}")
print(f"Test toplam zaman adimi: {sum(d['features'].shape[0] for d in test_items)}")


class SequenceDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]
        return d["features"], d["labels"]


def collate_fn(batch):
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    feats_padded = pad_sequence(feats, batch_first=True)                          # (B, Tmax, 768)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)  # (B, Tmax)
    return feats_padded, labels_padded, lengths


train_loader = DataLoader(SequenceDataset(train_items), batch_size=BATCH_SIZE,
                           shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(SequenceDataset(test_items), batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn)


# --------------------------------------------------------------------------
# Sinif agirligi (asiri buyuk agirliklari yumusatmak icin sqrt)
# --------------------------------------------------------------------------
all_train_labels = torch.cat([d["labels"] for d in train_items])
counts = torch.bincount(all_train_labels, minlength=num_classes).float()
class_weights = torch.sqrt(counts.sum() / (counts + 1e-6))
class_weights = (class_weights / class_weights.mean()).clamp(max=10.0).to(DEVICE)

print("En az ornekli 5 sinif (zaman adimi bazinda):")
for i in torch.argsort(counts)[:5]:
    print(f"  {IDX2LABEL[i.item()]}: {int(counts[i].item())} zaman adimi")


# --------------------------------------------------------------------------
# BiLSTM model
# --------------------------------------------------------------------------
class TemporalClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden=HIDDEN, n_classes=num_classes, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True,
                             bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        return self.head(out)  # (B, Tmax, n_classes)


model = TemporalClassifier().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

best_macro_f1 = 0.0
best_preds, best_true = None, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_steps = 0.0, 0
    for feats, labels, lengths in train_loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(feats, lengths)  # (B, Tmax, C)
        loss = criterion(logits.reshape(-1, num_classes), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        valid = (labels != IGNORE_INDEX).sum().item()
        total_loss += loss.item() * valid
        total_steps += valid
    scheduler.step()
    train_loss = total_loss / max(total_steps, 1)

    model.eval()
    epoch_preds, epoch_true = [], []
    with torch.no_grad():
        for feats, labels, lengths in test_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = model(feats, lengths)
            preds = logits.argmax(dim=-1)
            mask = labels != IGNORE_INDEX
            epoch_preds.append(preds[mask].cpu())
            epoch_true.append(labels[mask].cpu())
    epoch_preds = torch.cat(epoch_preds)
    epoch_true = torch.cat(epoch_true)
    test_acc = (epoch_preds == epoch_true).float().mean().item()
    macro_f1 = f1_score(epoch_true.numpy(), epoch_preds.numpy(),
                         labels=list(range(num_classes)), average="macro", zero_division=0)

    print(f"Epoch {epoch:2d}/{EPOCHS} - train_loss={train_loss:.4f}  "
          f"test_acc={test_acc:.4f}  macro_f1={macro_f1:.4f}")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_preds, best_true = epoch_preds, epoch_true
        torch.save(
            {"model_state": model.state_dict(), "label2idx": LABEL2IDX,
             "in_dim": 768, "hidden": HIDDEN, "n_classes": num_classes},
            SEQ_DIR / "temporal_model.pt",
        )

print(f"\nEn iyi macro F1: {best_macro_f1:.4f}")
print(f"Kaydedildi: {SEQ_DIR / 'temporal_model.pt'}")

target_names = [IDX2LABEL[i] for i in range(num_classes)]
print("\nSinif basina performans (en iyi epoch, zaman-adimi seviyesinde):")
print(classification_report(
    best_true.numpy(), best_preds.numpy(),
    labels=list(range(num_classes)), target_names=target_names, zero_division=0,
))
