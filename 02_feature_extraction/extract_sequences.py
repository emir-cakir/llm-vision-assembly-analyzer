"""
STAGE 3a (v2 - sekans tabanli) - Colab: her (video, kamera) icin klip
pencerelerini SIRALI olarak VideoMAE'den geciriyor ve bir "sekans" olarak
kaydediyor. Onceki yaklasimdan farki: klipler birbirinden bagimsiz,
karistirilmis bir havuz olarak degil, video ici SIRASIYLA saklaniyor -
boylece Stage 3b'de bir BiLSTM bu sirayi/baglami kullanabilecek.

tensors_fps8 (8 fps ile cikarilmis tensorler) uzerinde calisir.

Cikti: sequences/<video_key>__<cam>.pt
  { "features": (T_windows, 768) float32,
    "labels":   (T_windows,) long  (kaba/coarse etiket index'i),
    "subset":   "training" / "testing",
    "video_key": str, "cam": str,
    "center_original_frames": (T_windows,) - her pencerenin orijinal
        (native-fps) merkez karesi, rapor asamasinda zamana cevirmek icin }

Tahmini sure: fps=2 surumune gore ~4x daha yogun pencere var (fps artisi)
STRIDE=8 (8fps'te =1 saniyelik adim) ile makul bir denge.
"""

import subprocess
import sys

try:
    import transformers  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
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
GT_SEGMENTS_PATH = DRIVE_ROOT / "gt_segments.json"

CLIP_LEN = 16      # VideoMAE-base'in sabit gereksinimi
STRIDE = 8         # 8fps'te = 1 saniyelik adim
CAMERAS = ["dev1", "dev2", "dev3"]
BATCH_SIZE = 16    # ayni video icindeki pencereleri toplu halde modele veriyoruz

OUT_DIR = DRIVE_ROOT / "sequences"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Kullanilan cihaz:", DEVICE)

# --------------------------------------------------------------------------
# gt_segments.json + metadata.json
# --------------------------------------------------------------------------
with open(GT_SEGMENTS_PATH, "r", encoding="utf-8") as f:
    gt = json.load(f)["database"]

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Ayni kaba (coarse) etiket eslemesi - train_classifier.py'deki ile AYNI
COARSE_MAP = {
    "NA": "NA", "other": "other",
    "align leg screw with table thread": "align part",
    "align side panel holes with front panel dowels": "align part",
    "attach drawer back panel": "attach part", "attach drawer side panel": "attach part",
    "attach shelf to table": "attach part", "insert drawer pin": "attach part",
    "flip shelf": "flip part", "flip table": "flip part", "flip table top": "flip part",
    "position the drawer right side up": "flip part",
    "lay down back panel": "lay down part", "lay down bottom panel": "lay down part",
    "lay down front panel": "lay down part", "lay down leg": "lay down part",
    "lay down shelf": "lay down part", "lay down side panel": "lay down part",
    "lay down table top": "lay down part",
    "pick up back panel": "pick up part", "pick up bottom panel": "pick up part",
    "pick up front panel": "pick up part", "pick up leg": "pick up part",
    "pick up pin": "pick up part", "pick up shelf": "pick up part",
    "pick up side panel": "pick up part", "pick up table top": "pick up part",
    "push table": "push table", "push table top": "push table",
    "rotate table": "rotate table", "slide bottom of drawer": "slide part",
    "spin leg": "spin leg", "tighten leg": "tighten leg",
}
LABEL2IDX = {label: i for i, label in enumerate(sorted(set(COARSE_MAP.values())))}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}
print(f"Toplam {len(LABEL2IDX)} kaba etiket: {list(LABEL2IDX.keys())}")

with open(OUT_DIR / "label_vocab.json", "w", encoding="utf-8") as f:
    json.dump(LABEL2IDX, f, ensure_ascii=False, indent=2)


def majority_label_for_window(annotations, original_frame_indices):
    labels = []
    for f in original_frame_indices:
        for seg in annotations:
            s, e = seg["segment"]
            if s <= f <= e:
                labels.append(COARSE_MAP[seg["label"]])
                break
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


# --------------------------------------------------------------------------
# Frozen VideoMAE-base
# --------------------------------------------------------------------------
print("VideoMAE-base yukleniyor...")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
model.eval().to(DEVICE)
for p in model.parameters():
    p.requires_grad_(False)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


@torch.no_grad()
def extract_windows_for_video(npy_path, starts):
    """Bir video-kamera icin verilen tum pencereleri batch'ler halinde
    VideoMAE'den gecirir, (N, 768) feature dondurur."""
    arr = np.load(npy_path, mmap_mode="r")
    all_feats = []
    for i in range(0, len(starts), BATCH_SIZE):
        batch_starts = starts[i:i + BATCH_SIZE]
        clips = np.stack([np.array(arr[s:s + CLIP_LEN]) for s in batch_starts])  # (B,T,H,W,C)
        clips = torch.from_numpy(clips).float() / 255.0
        clips = clips.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        clips = (clips - IMAGENET_MEAN) / IMAGENET_STD
        clips = clips.permute(0, 2, 1, 3, 4).to(DEVICE)  # VideoMAE bekledigi: (B, T, C, H, W)

        outputs = model(pixel_values=clips)
        feats = outputs.last_hidden_state.mean(dim=1)  # (B, 768)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


# --------------------------------------------------------------------------
# Ana dongu: her video-kamera icin sirali pencere dizisi olustur ve kaydet
# --------------------------------------------------------------------------
video_keys = list(gt.keys())
n_done, n_skip, n_empty = 0, 0, 0

for vi, video_key in enumerate(video_keys):
    entry = gt[video_key]
    subset = entry["subset"]
    annotations = entry["annotation"]
    video_meta = metadata.get(video_key)
    if not video_meta:
        continue

    for cam in CAMERAS:
        cam_meta = video_meta.get(cam)
        if not cam_meta:
            continue

        out_name = f"{video_key.replace('/', '__')}__{cam}.pt"
        out_path = OUT_DIR / out_name
        if out_path.exists():
            n_skip += 1
            continue

        npy_path = TENSORS_DIR / video_key.replace("/", "__") / f"{cam}.npy"
        if not npy_path.exists():
            continue

        sampled_idx = cam_meta["sampled_original_frame_indices"]
        T = cam_meta["tensor_num_frames"]
        if T is None or T != len(sampled_idx) or T < CLIP_LEN:
            continue

        starts = list(range(0, T - CLIP_LEN + 1, STRIDE))
        if not starts:
            n_empty += 1
            continue

        labels, center_frames = [], []
        for s in starts:
            window_frames = sampled_idx[s:s + CLIP_LEN]
            label = majority_label_for_window(annotations, window_frames)
            labels.append(LABEL2IDX[label] if label is not None else LABEL2IDX["NA"])
            center_frames.append(window_frames[len(window_frames) // 2])

        feats = extract_windows_for_video(npy_path, starts)

        torch.save({
            "features": feats,
            "labels": torch.tensor(labels, dtype=torch.long),
            "subset": subset,
            "video_key": video_key,
            "cam": cam,
            "center_original_frames": torch.tensor(center_frames, dtype=torch.long),
        }, out_path)

        n_done += 1
        if n_done % 25 == 0:
            print(f"[{vi + 1}/{len(video_keys)}] {n_done} sekans kaydedildi "
                  f"(son: {video_key} [{cam}], {len(starts)} pencere)")

print(f"\nBitti. Kaydedilen: {n_done}, atlanan(zaten var): {n_skip}, bos: {n_empty}")
