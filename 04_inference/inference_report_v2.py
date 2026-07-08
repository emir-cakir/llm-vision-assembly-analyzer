"""
STAGE 4a (v2) - iki iyilestirme eklendi:
  1) COKLU KAMERA BIRLESTIRME: ayni video icin mevcut tum kameralarin
     (dev1/dev2/dev3) tahminlerini zaman bazinda hizalayip oy coklugu ile
     birlestiriyoruz -> daha guvenilir tek bir tahmin dizisi.
  2) MOBILYA-TIPINE OZEL STANDART SURE: "standart sure" artik tum veri
     setinin degil, o videonun mobilya kategorisinin (orn. Lack_TV_Bench)
     medyanindan hesaplaniyor; o kategori icin yeterli veri yoksa genel
     medyana geri duser.

Eski inference_report.py (tek kamera, genel standart) dokunulmadan duruyor -
bu script ayri, karsilastirma yapmak istersen ikisini de kullanabilirsin.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import VideoMAEModel

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
TENSORS_DIR = Path("/content/drive/Othercomputers/Dizüstü Bilgisayarım/tensors_fps8")
METADATA_PATH = TENSORS_DIR / "metadata.json"
SEQ_DIR = DRIVE_ROOT / "sequences"
MODEL_PATH = SEQ_DIR / "temporal_model.pt"

CLIP_LEN = 16
STRIDE = 8   # extract_sequences.py ile AYNI olmali
CAMERAS = ["dev1", "dev2", "dev3"]

# Analiz edilecek video - istedigin baska bir video_key ile degistirebilirsin
VIDEO_KEY = "Lack_TV_Bench/0025_black_table_04_02_2019_08_20_13_48"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Kullanilan cihaz:", DEVICE)

# --------------------------------------------------------------------------
# Mobilya-tipine ozel + genel standart sureler (egitim verisinden hesaplanmis)
# --------------------------------------------------------------------------
STANDARD_DURATIONS = {
    "global": {
        "NA": {"median_sec": 1.24, "n": 2598}, "flip part": {"median_sec": 5.04, "n": 433},
        "handle part": {"median_sec": 1.52, "n": 1922}, "align part": {"median_sec": 1.76, "n": 1032},
        "spin leg": {"median_sec": 15.36, "n": 1116}, "other": {"median_sec": 2.92, "n": 80},
        "tighten leg": {"median_sec": 3.0, "n": 329}, "move table": {"median_sec": 2.36, "n": 217},
        "attach part": {"median_sec": 5.6, "n": 597}, "slide part": {"median_sec": 6.92, "n": 89},
    },
    "by_furniture": {
        "Lack_TV_Bench": {
            "NA": {"median_sec": 1.12, "n": 671}, "flip part": {"median_sec": 4.68, "n": 124},
            "handle part": {"median_sec": 1.56, "n": 470}, "align part": {"median_sec": 1.72, "n": 305},
            "spin leg": {"median_sec": 15.08, "n": 365}, "other": {"median_sec": 2.48, "n": 24},
            "tighten leg": {"median_sec": 3.12, "n": 123}, "move table": {"median_sec": 1.96, "n": 53},
            "attach part": {"median_sec": 11.24, "n": 91},
        },
        "Lack_Coffee_Table": {
            "NA": {"median_sec": 1.6, "n": 678}, "handle part": {"median_sec": 1.72, "n": 467},
            "flip part": {"median_sec": 6.4, "n": 131}, "align part": {"median_sec": 1.76, "n": 275},
            "spin leg": {"median_sec": 15.16, "n": 372}, "attach part": {"median_sec": 16.24, "n": 93},
            "tighten leg": {"median_sec": 3.3, "n": 94}, "move table": {"median_sec": 3.04, "n": 57},
            "other": {"median_sec": 3.68, "n": 20},
        },
        "Lack_Side_Table": {
            "NA": {"median_sec": 1.08, "n": 588}, "flip part": {"median_sec": 4.94, "n": 116},
            "align part": {"median_sec": 1.6, "n": 311}, "spin leg": {"median_sec": 16.0, "n": 379},
            "handle part": {"median_sec": 1.36, "n": 384}, "move table": {"median_sec": 2.2, "n": 107},
            "tighten leg": {"median_sec": 2.64, "n": 112}, "other": {"median_sec": 3.04, "n": 9},
        },
        "Kallax_Shelf_Drawer": {
            "NA": {"median_sec": 1.4, "n": 661}, "handle part": {"median_sec": 1.52, "n": 601},
            "align part": {"median_sec": 2.8, "n": 141}, "attach part": {"median_sec": 3.28, "n": 413},
            "slide part": {"median_sec": 6.92, "n": 89}, "flip part": {"median_sec": 3.04, "n": 62},
            "other": {"median_sec": 2.84, "n": 27},
        },
    },
}


def get_standard(furniture: str, label: str):
    """Once mobilya-ozel medyani dener, yoksa genel medyana duser."""
    by_furn = STANDARD_DURATIONS["by_furniture"].get(furniture, {})
    if label in by_furn:
        return by_furn[label]["median_sec"], "furniture"
    if label in STANDARD_DURATIONS["global"]:
        return STANDARD_DURATIONS["global"][label]["median_sec"], "global"
    return None, None


# --------------------------------------------------------------------------
# Frozen VideoMAE + egitilmis BiLSTM (bir kere yukle, tum kameralar icin kullan)
# --------------------------------------------------------------------------
print("VideoMAE-base yukleniyor...")
videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").eval().to(DEVICE)
for p in videomae.parameters():
    p.requires_grad_(False)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


class TemporalClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden=256, n_classes=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        return self.head(out)


ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
label2idx = ckpt["label2idx"]
idx2label = {v: k for k, v in label2idx.items()}

bilstm = TemporalClassifier(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"],
                             n_classes=ckpt["n_classes"]).to(DEVICE)
bilstm.load_state_dict(ckpt["model_state"])
bilstm.eval()

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


def predict_camera(video_key: str, cam: str):
    """Tek bir kamera icin (zaman_saniye -> tahmin_edilen_etiket) sozlugu dondurur."""
    video_meta = metadata.get(video_key, {})
    cam_meta = video_meta.get(cam)
    npy_path = TENSORS_DIR / video_key.replace("/", "__") / f"{cam}.npy"
    if not cam_meta or not npy_path.exists():
        return None

    sampled_idx = cam_meta["sampled_original_frame_indices"]
    native_fps = cam_meta["native_fps"]
    target_fps = cam_meta["target_fps"]
    T = cam_meta["tensor_num_frames"]

    arr = np.load(npy_path, mmap_mode="r")
    starts = list(range(0, T - CLIP_LEN + 1, STRIDE))
    if not starts:
        return None
    center_frames = [sampled_idx[s + CLIP_LEN // 2] for s in starts]

    BATCH = 16
    feats = []
    with torch.no_grad():
        for i in range(0, len(starts), BATCH):
            b = starts[i:i + BATCH]
            clips = np.stack([np.array(arr[s:s + CLIP_LEN]) for s in b])
            clips = torch.from_numpy(clips).float() / 255.0
            clips = clips.permute(0, 4, 1, 2, 3)
            clips = (clips - IMAGENET_MEAN) / IMAGENET_STD
            clips = clips.permute(0, 2, 1, 3, 4).to(DEVICE)
            out = videomae(pixel_values=clips).last_hidden_state.mean(dim=1)
            feats.append(out.cpu())
    feats = torch.cat(feats, dim=0)

    with torch.no_grad():
        x = feats.unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([feats.shape[0]])
        logits = bilstm(x, lengths)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    window_step_sec = STRIDE / target_fps
    time_to_label = {}
    for i, center_f in enumerate(center_frames):
        t = round(center_f / native_fps)   # en yakin saniyeye yuvarla (birlestirme icin ortak eksen)
        time_to_label[t] = idx2label[preds[i]]
    return time_to_label


# --------------------------------------------------------------------------
# Mevcut tum kameralari isle, zaman bazinda oy cokluguyla birlestir
# --------------------------------------------------------------------------
per_camera_predictions = {}
for cam in CAMERAS:
    result = predict_camera(VIDEO_KEY, cam)
    if result is not None:
        per_camera_predictions[cam] = result
        print(f"{cam}: {len(result)} zaman noktasi tahmin edildi")
    else:
        print(f"{cam}: bulunamadi/atlandi")

if not per_camera_predictions:
    raise RuntimeError(f"{VIDEO_KEY} icin hicbir kamerada veri bulunamadi.")

all_times = sorted(set().union(*[set(d.keys()) for d in per_camera_predictions.values()]))

fused = {}
agreement = {}
for t in all_times:
    votes = [d[t] for d in per_camera_predictions.values() if t in d]
    counter = Counter(votes)
    top_label, top_count = counter.most_common(1)[0]
    fused[t] = top_label
    agreement[t] = top_count / len(votes)   # o zaman noktasinda kameralarin ne kadari hemfikir

print(f"\n{len(per_camera_predictions)} kamera birlestirildi, {len(all_times)} zaman noktasi.")

# --------------------------------------------------------------------------
# Zaman siralamasindan segmentlere birlestir
# --------------------------------------------------------------------------
furniture = VIDEO_KEY.split("/")[0]

segments = []
cur_label, cur_start_t = fused[all_times[0]], all_times[0]
cur_agree = [agreement[all_times[0]]]
for t in all_times[1:]:
    if fused[t] != cur_label:
        segments.append((cur_label, cur_start_t, t, cur_agree))
        cur_label, cur_start_t, cur_agree = fused[t], t, []
    cur_agree.append(agreement[t])
segments.append((cur_label, cur_start_t, all_times[-1] + 1, cur_agree))

report_segments = []
for label, start_t, end_t, agree_list in segments:
    duration = end_t - start_t
    std_median, std_source = get_standard(furniture, label)
    pct_diff = None
    if std_median and std_median > 0:
        pct_diff = round((duration - std_median) / std_median * 100, 1)

    report_segments.append({
        "label": label,
        "start_sec": start_t,
        "end_sec": end_t,
        "duration_sec": duration,
        "standard_median_sec": std_median,
        "standard_source": std_source,   # "furniture" veya "global"
        "pct_diff_vs_standard": pct_diff,
        "camera_agreement": round(sum(agree_list) / len(agree_list), 2),
    })

print(f"\n--- Tahmin Edilen Segmentler ({furniture}, {len(per_camera_predictions)} kamera birlesimi) ---")
for seg in report_segments:
    diff_str = f"({seg['pct_diff_vs_standard']:+.0f}%)" if seg["pct_diff_vs_standard"] is not None else ""
    print(f"[{seg['start_sec']:4d}s-{seg['end_sec']:4d}s] {seg['label']:<15} "
          f"sure={seg['duration_sec']:3d}s standart={seg['standard_median_sec']}s[{seg['standard_source']}] "
          f"{diff_str} uyum={seg['camera_agreement']}")

out_data = {
    "video_key": VIDEO_KEY,
    "furniture": furniture,
    "cameras_used": list(per_camera_predictions.keys()),
    "segments": report_segments,
}
out_path = DRIVE_ROOT / "video_analysis_report_multicam.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)

print(f"\nKaydedildi: {out_path}")
