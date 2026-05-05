#!/usr/bin/env python3
"""
Process a video with SAM2 to extract per-frame body masks.

Supports multi-person: uses YOLOv8 to auto-detect people in frame 0,
prompts SAM2 with a bounding box per person, and saves masks with
pixel values = person ID (1, 2, ..., N). Single-person videos get
binary masks (0/1) for backward compatibility.

Usage:
    python process_video_sam2.py <video_path> --name <name> [--max-frames 600]

Output structure (in research/videos/<name>/):
    frames/   - JPEG frames (00000.jpg, 00001.jpg, ...)
    masks/    - PNG masks with person IDs (00000.png, 00001.png, ...)
    config.json - metadata including numPersons
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def extract_frames(video_path, output_dir, max_frames=0):
    """Extract JPEG frames from a video file.

    max_frames=0 means extract all frames (no limit).
    """
    print("Extracting frames...", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames > 0 and total_raw > max_frames:
        indices = np.linspace(0, total_raw - 1, max_frames, dtype=int)
    else:
        indices = np.arange(total_raw)

    idx_set = set(indices.tolist())
    frame_idx = 0
    out_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in idx_set:
            out_path = os.path.join(output_dir, f"{out_idx:05d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            out_idx += 1
            written += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {written} frames ({width}x{height} @ {fps:.1f} fps)", flush=True)
    return written, fps, width, height


def detect_persons(frame_path, min_confidence=0.5, min_area_ratio=0.01):
    """Detect persons in a frame using YOLOv8.

    Returns list of (x1, y1, x2, y2) bounding boxes sorted left-to-right,
    or falls back to a single center-body prompt if YOLO is unavailable.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("YOLOv8 not available, falling back to single-person center prompt", flush=True)
        return None

    model = YOLO("yolov8n.pt")  # nano model, auto-downloads
    results = model(frame_path, verbose=False)

    frame = cv2.imread(frame_path)
    h, w = frame.shape[:2]
    min_area = w * h * min_area_ratio

    boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf >= min_confidence:  # class 0 = person
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area >= min_area:
                    boxes.append((x1, y1, x2, y2))

    if not boxes:
        print("No persons detected by YOLO, falling back to center prompt", flush=True)
        return None

    # Sort left-to-right by center x for consistent person IDs
    boxes.sort(key=lambda b: (b[0] + b[2]) / 2)
    print(f"Detected {len(boxes)} person(s) via YOLO", flush=True)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        print(f"  Person {i+1}: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})", flush=True)
    return boxes


def body_prompt_points(width, height):
    """Generate point prompts targeting a standing person in the frame center.

    Fallback when YOLO is not available or detects no persons.
    Returns (points, labels) as numpy arrays.
    """
    cx = width / 2

    fg_points = [
        (cx, height * 0.35),
        (cx, height * 0.45),
        (cx, height * 0.55),
        (cx, height * 0.65),
        (cx, height * 0.75),
    ]

    margin = min(width, height) * 0.05
    bg_points = [
        (cx, margin),
        (margin, margin),
        (width - margin, margin),
        (margin, height - margin),
        (width - margin, height - margin),
    ]

    points = np.array(fg_points + bg_points, dtype=np.float32)
    labels = np.array(
        [1] * len(fg_points) + [0] * len(bg_points),
        dtype=np.int32,
    )

    print(f"Body prompt: {len(fg_points)} fg + {len(bg_points)} bg points", flush=True)
    return points, labels


def run_sam2(frames_dir, masks_dir, num_frames, width, height):
    """Run SAM2 video predictor to generate per-person body masks."""
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    print("Running SAM2 video predictor...", flush=True)
    os.makedirs(masks_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "facebook/sam2.1-hiera-large"
    print(f"Loading model: {model_id}", flush=True)

    predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)

    print("Initializing video state...", flush=True)
    inference_state = predictor.init_state(
        video_path=frames_dir,
        offload_video_to_cpu=True,
    )

    # Detect persons in frame 0
    frame0_path = os.path.join(frames_dir, "00000.jpg")
    person_boxes = detect_persons(frame0_path)

    if person_boxes and len(person_boxes) >= 1:
        # Multi-person: prompt SAM2 with a bounding box per person
        num_persons = len(person_boxes)
        for i, (x1, y1, x2, y2) in enumerate(person_boxes):
            obj_id = i + 1  # 1-based person IDs
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                box=box,
            )
            print(f"  Added box prompt for person {obj_id}", flush=True)
    else:
        # Single-person fallback: point prompts on center body
        num_persons = 1
        points, labels = body_prompt_points(width, height)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels,
        )

    # Propagate through all frames
    # Each frame yields (frame_idx, obj_ids, mask_logits)
    # mask_logits shape: (num_objects, 1, H, W)
    all_masks = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        # Combine per-object masks into a single instance mask
        # pixel value = person ID (1-N), 0 = background
        h_out, w_out = mask_logits.shape[-2:]
        combined = np.zeros((h_out, w_out), dtype=np.uint8)
        for k, oid in enumerate(obj_ids):
            person_mask = (mask_logits[k, 0] > 0.0).cpu().numpy()
            combined[person_mask] = int(oid)
        all_masks[frame_idx] = combined
        print(f"  frame {frame_idx + 1}/{num_frames}", flush=True)

    # Save masks as PNGs
    for i in range(num_frames):
        mask_path = os.path.join(masks_dir, f"{i:05d}.png")
        if i in all_masks:
            mask = all_masks[i]
            if mask.shape[0] != height or mask.shape[1] != width:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            # Single person: scale to 0/255 for backward compat with old viewer
            if num_persons == 1:
                mask = mask * 255
            cv2.imwrite(mask_path, mask)
        else:
            cv2.imwrite(mask_path, np.zeros((height, width), dtype=np.uint8))

    # Clean up GPU memory
    predictor.reset_state(inference_state)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Saved {len(all_masks)} masks ({num_persons} person(s)) to {masks_dir}", flush=True)
    return num_persons


def main():
    parser = argparse.ArgumentParser(description="Process video with SAM2")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--name", required=True, help="Output directory name")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0 = all)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    videos_dir = base_dir / "research" / "videos"
    output_dir = videos_dir / args.name
    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"

    # Step 1: Extract frames
    num_frames, fps, width, height = extract_frames(
        args.video_path, str(frames_dir), args.max_frames
    )

    # Step 2: Run SAM2
    num_persons = run_sam2(str(frames_dir), str(masks_dir), num_frames, width, height)

    # Step 3: Write config
    config = {
        "name": args.name,
        "totalFrames": num_frames,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "numPersons": num_persons,
        "maskDir": "masks/",
        "frameDir": "frames/",
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config written to {config_path}", flush=True)
    print("=== Done ===", flush=True)


if __name__ == "__main__":
    main()
