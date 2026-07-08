r"""
Her (video, kamera) cifti icin native fps, toplam orijinal kare sayisi ve
tensore ornekleme sirasinda SECILEN orijinal kare indekslerini hesaplayip
tek bir metadata.json dosyasina yazar.

Bu dosya olmadan, .npy tensorlerindeki kare 5 ile gt_segments.json'daki
"segment": [start, end] (orijinal, native-fps kare numaralari) arasinda
dogru eslesme kurulamaz -> Stage 2'de etiketler yanlis hizalanir.

video_to_tensor.py ile AYNI --fps, --size ve --cameras degerlerini
vermelisin, yoksa hesaplanan indeksler gercek tensorle uyusmaz.

Kullanim:
    python build_metadata.py --root "D:\videos" --gt_json gt_segments.json \
                              --tensors_dir .\tensors --fps 2 \
                              --out .\tensors\metadata.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def sample_indices(total_frames: int, native_fps: float, target_fps: float):
    """video_to_tensor.py'deki fonksiyonla BIREBIR ayni olmali."""
    if target_fps >= native_fps or native_fps <= 0:
        return list(range(total_frames))
    step = native_fps / target_fps
    idxs = []
    i = 0.0
    while int(i) < total_frames:
        idxs.append(int(i))
        i += step
    return idxs


def find_camera_files(video_dir: Path, cam: str):
    exact = video_dir / cam / "images" / "scan_video.avi"
    if exact.exists():
        return [exact]
    cam_dir = video_dir / cam
    matches = list(cam_dir.rglob("*.avi")) if cam_dir.exists() else []
    if not matches:
        matches = list(cam_dir.rglob("*.mp4")) if cam_dir.exists() else []
    if not matches:
        matches = list(video_dir.rglob(f"*{cam}*.avi")) + list(video_dir.rglob(f"*{cam}*.mp4"))
    return matches


def npy_shape_fast(path: Path):
    """Tum diziyi RAM'e kopyalamadan (memory-map ile) sadece shape okur."""
    arr = np.load(path, mmap_mode="r")
    return arr.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--gt_json", required=True)
    parser.add_argument("--tensors_dir", required=True)
    parser.add_argument("--fps", type=float, default=2.0,
                         help="video_to_tensor.py'de kullandigin --fps ile AYNI olmali")
    parser.add_argument("--cameras", nargs="+", default=["dev1", "dev2", "dev3"])
    parser.add_argument("--assume_fps", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt = json.load(f)

    root = Path(args.root)
    tensors_dir = Path(args.tensors_dir)
    video_keys = list(gt["database"].keys())

    metadata = {}
    n_ok, n_missing_npy, n_mismatch, n_no_video = 0, 0, 0, 0

    for key in video_keys:
        video_dir = root / key
        for cam in args.cameras:
            npy_path = tensors_dir / key.replace("/", "__") / f"{cam}.npy"
            if not npy_path.exists():
                continue  # bu kamera zaten yoktu (Stage 1'de "bulunamadi" idi)

            candidates = find_camera_files(video_dir, cam)
            if not candidates:
                n_no_video += 1
                print(f"UYARI: {npy_path} var ama kaynak video bulunamadi ({video_dir / cam})")
                continue
            video_path = candidates[0]

            cap = cv2.VideoCapture(str(video_path))
            native_fps = args.assume_fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            sampled = sample_indices(total_frames, native_fps, args.fps)

            # tensorun gercek uzunuguyla karsilastir (tutarlilik kontrolu)
            try:
                shape = npy_shape_fast(npy_path)
                tensor_len = shape[0]
            except Exception as e:
                print(f"UYARI: {npy_path} shape okunamadi: {e}")
                tensor_len = None

            if tensor_len is not None and tensor_len != len(sampled):
                n_mismatch += 1
                print(f"UYUSMAZLIK [{key} {cam}]: tensor={tensor_len} kare, "
                      f"hesaplanan={len(sampled)} kare -> bu videoyu tekrar incelemek gerekebilir")

            metadata.setdefault(key, {})[cam] = {
                "native_fps": native_fps,
                "total_frames_original": total_frames,
                "target_fps": args.fps,
                "sampled_original_frame_indices": sampled,
                "tensor_num_frames": tensor_len,
            }
            n_ok += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print(f"\nBitti. Islenen (video,kamera): {n_ok}, "
          f"kaynak video bulunamadi: {n_no_video}, "
          f"uzunluk uyusmazligi: {n_mismatch}")
    print(f"Metadata yazildi: {args.out}")


if __name__ == "__main__":
    main()
