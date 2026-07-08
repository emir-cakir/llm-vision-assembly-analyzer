r"""
IKEA ASM videolarini (dev*/images/scan_video.avi) Colab'a yuklemeden once
yerel bilgisayarda kucuk, dusuk-fps tensorlere (.npy, uint8) donusturur.

PARALEL (multiprocessing) versiyon: birden fazla videoyu ayni anda,
farkli CPU cekirdeklerinde isler.

Kullanim:
    python video_to_tensor.py --root "D:\videos" \
                               --gt_json gt_segments.json \
                               --out ./tensors \
                               --fps 2 --size 224 --workers 8

Beklenen klasor yapisi:
    <root>/<furniture>/<video_id>/<dev1|dev2|dev3>/images/scan_video.avi
  ornek: D:\videos\Kallax_Shelf_Drawer\0001_black_table_02_01_2019_08_16_14_00\dev1\images\scan_video.avi
  gt_segments.json'daki key: "Kallax_Shelf_Drawer/0001_black_table_02_01_2019_08_16_14_00"

Notlar:
- --workers verilmezse otomatik olarak (CPU cekirdek sayisi - 2) kullanilir,
  boylece isletim sistemi icin de biraz pay birakilir.
- Zaten var olan .npy dosyalari atlanir -> script yarida kesilirse
  tekrar calistirinca kaldigi yerden devam eder.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def sample_indices(total_frames: int, native_fps: float, target_fps: float):
    """Native fps'ten target fps'e esit araliklarla frame index'i secer."""
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
    # Bilinen IKEA ASM yapisi: <video_dir>/<cam>/images/scan_video.avi
    exact = video_dir / cam / "images" / "scan_video.avi"
    if exact.exists():
        return [exact]

    # Uzanti/isim farkli olabilir diye esnek arama (avi ve mp4 ikisini de dene)
    cam_dir = video_dir / cam
    matches = list(cam_dir.rglob("*.avi")) if cam_dir.exists() else []
    if not matches:
        matches = list(cam_dir.rglob("*.mp4")) if cam_dir.exists() else []
    if not matches:
        # en son care: video_dir altinda kamera adiyla herhangi bir yerde ara
        matches = list(video_dir.rglob(f"*{cam}*.avi")) + list(video_dir.rglob(f"*{cam}*.mp4"))
    return matches


def process_video(mp4_path: Path, out_path: Path, target_fps: float, size: int,
                   assume_fps: float = None):
    cap = cv2.VideoCapture(str(mp4_path))
    native_fps = assume_fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(sample_indices(total, native_fps, target_fps))

    frames = []
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no in wanted:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_no += 1
    cap.release()

    if not frames:
        return None

    arr = np.stack(frames).astype(np.uint8)  # (T, H, W, 3)
    np.save(out_path, arr)
    return arr.shape


def _worker(task):
    """ProcessPoolExecutor icin picklable top-level fonksiyon."""
    key, cam, mp4_path_str, out_file_str, target_fps, size, assume_fps = task
    mp4_path = Path(mp4_path_str)
    out_file = Path(out_file_str)
    try:
        shape = process_video(mp4_path, out_file, target_fps, size, assume_fps)
    except Exception as e:
        return key, cam, "error", str(e)
    if shape is None:
        return key, cam, "error", "hic frame okunamadi (bozuk dosya olabilir)"
    return key, cam, "ok", shape


def build_tasks(root: Path, out_root: Path, video_keys, cameras, target_fps, size, assume_fps):
    tasks = []
    n_skip = 0
    n_missing = 0
    for key in video_keys:
        video_dir = root / key
        if not video_dir.exists():
            print(f"KLASOR YOK: {video_dir}")
            n_missing += 1
            continue

        for cam in cameras:
            candidates = find_camera_files(video_dir, cam)
            if not candidates:
                print(f"  [{key}] {cam} bulunamadi: {video_dir / cam}")
                n_missing += 1
                continue
            mp4_path = candidates[0]

            out_dir = out_root / key.replace("/", "__")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{cam}.npy"

            if out_file.exists():
                n_skip += 1
                continue

            tasks.append((key, cam, str(mp4_path), str(out_file), target_fps, size, assume_fps))

    return tasks, n_skip, n_missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Videolarin bulundugu kok klasor")
    parser.add_argument("--gt_json", required=True, help="gt_segments.json yolu")
    parser.add_argument("--out", required=True, help="Cikti tensorlerinin yazilacagi klasor")
    parser.add_argument("--fps", type=float, default=2.0, help="Hedef ornekleme fps'i")
    parser.add_argument("--size", type=int, default=224, help="Kare boyutu (size x size)")
    parser.add_argument("--cameras", nargs="+", default=["dev1", "dev2", "dev3"])
    parser.add_argument("--assume_fps", type=float, default=None,
                         help="cv2 native fps okuyamazsa elle vermek icin")
    parser.add_argument("--limit", type=int, default=None,
                         help="Test icin sadece ilk N videoyu isle")
    parser.add_argument("--workers", type=int, default=None,
                         help="Paralel islem sayisi (varsayilan: CPU cekirdek sayisi - 2)")
    args = parser.parse_args()

    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)

    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt = json.load(f)

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    video_keys = list(gt["database"].keys())
    if args.limit:
        video_keys = video_keys[: args.limit]
    print(f"{len(video_keys)} video taranacak, {workers} paralel islem kullanilacak.")

    tasks, n_skip, n_missing = build_tasks(
        root, out_root, video_keys, args.cameras, args.fps, args.size, args.assume_fps
    )
    print(f"{len(tasks)} yeni gorev islenecek ({n_skip} zaten mevcut, {n_missing} bulunamadi/eksik).")

    n_ok, n_fail = 0, 0
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker, t) for t in tasks]
            done = 0
            total = len(futures)
            for fut in as_completed(futures):
                key, cam, status, info = fut.result()
                done += 1
                if status == "ok":
                    n_ok += 1
                    print(f"[{done}/{total}] {key} [{cam}] -> {info}")
                else:
                    n_fail += 1
                    print(f"[{done}/{total}] HATA {key} [{cam}]: {info}")

    print(f"\nBitti. Basarili: {n_ok}, atlanan(zaten var): {n_skip}, "
          f"bulunamadi: {n_missing}, hata: {n_fail}")


if __name__ == "__main__":
    main()
