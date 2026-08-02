#!/usr/bin/env python3
"""
Unified Sanity-check for Extracted Keypoints (EgoGesture & Jester).

Outputs:
- summary.json
- per_file_metrics.csv
- overlays/{worst,best,random}/*.png
- contact_sheets/{worst,best,random}_contact_sheet.png

How to run:
    python tools/data_prep/sanity_check.py \
    --keypoints-dir dataset/EgoGesture/keypoints_mediapipe \
    --output-dir dataset/EgoGesture/sanity_report \
    --base-dir dataset/EgoGesture/

    python tools/data_prep/sanity_check.py \
    --keypoints-dir dataset/Jester/keypoints_mediapipe \
    --output-dir dataset/Jester/sanity_report \
    --base-dir dataset/Jester/20bn-jester-v1/
"""

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import cv2
import numpy as np

# Standard hand keypoint connections (MediaPipe)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]

def parse_args():
    parser = argparse.ArgumentParser(description="Sanity-check extracted keypoints")
    parser.add_argument('--keypoints-dir', required=True, type=str,
                        help='Directory with PKL files')
    parser.add_argument('--output-dir', required=True, type=str,
                        help='Output directory for reports and images')
    parser.add_argument('--base-dir', default='.', type=str,
                        help='Base directory to resolve relative video paths (e.g. raw video folder)')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--visuals-per-group', default=12, type=int,
                        help='How many samples for best/worst/random overlays')
    parser.add_argument('--thumb-width', default=700, type=int,
                        help='Thumbnail width for contact sheets')

    # Quality thresholds
    parser.add_argument('--thr-valid-good', default=0.70, type=float,
                        help='Good if valid_ratio >= this threshold')
    parser.add_argument('--thr-valid-bad', default=0.45, type=float,
                        help='Bad if valid_ratio < this threshold')
    parser.add_argument('--thr-oob-bad', default=0.10, type=float,
                        help='Bad if out-of-bounds ratio > this threshold')
    parser.add_argument('--thr-motion-low', default=2.0, type=float,
                        help='Suspiciously low wrist motion (pixels/frame)')
    return parser.parse_args()


def resolve_video_path(data, base_dir):
    base_dir = Path(base_dir)
    
    # Try EgoGesture logic first
    if 'video_path' in data and data['video_path']:
        p = Path(data['video_path'])
        if p.exists(): return p
        p2 = (base_dir / data['video_path']).resolve()
        if p2.exists(): return p2
        
    # Try Jester logic (video_id is the folder name)
    if 'video_id' in data:
        p_jester = (base_dir / str(data['video_id'])).resolve()
        if p_jester.exists() and p_jester.is_dir():
            return p_jester
            
    return None


def read_frame(video_path, frame_idx):
    video_path = Path(video_path)
    
    # Handle Jester (Directory of frames)
    if video_path.is_dir():
        frames = sorted(video_path.glob('*.jpg'))
        if not frames:
            return None, None, None
        total = len(frames)
        frame_idx = int(np.clip(frame_idx, 0, max(0, total - 1)))
        
        frame = cv2.imread(str(frames[frame_idx]))
        if frame is None:
            return None, None, None
        return frame, frame.shape[1], frame.shape[0]

    # Handle EgoGesture (Video file)
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None, None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = int(np.clip(frame_idx, 0, max(0, total - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            return None, w, h
        return frame, w, h


def draw_overlay(frame, keypoints, bbox=None):
    out = frame.copy()
    pts = keypoints.astype(np.int32)

    for i, j in CONNECTIONS:
        cv2.line(out, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(out, tuple(p), 3, (0, 0, 255), -1, cv2.LINE_AA)

    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = out.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 2, cv2.LINE_AA)

    return out


def draw_panel_title(image, title):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 40), (245, 245, 245), -1)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (180, 180, 180), 1)
    cv2.putText(out, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return out


def make_panel(image, title, panel_w=500, panel_h=340):
    head_h = 44
    body_h = panel_h - head_h
    panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)

    h, w = image.shape[:2]
    scale = min(panel_w / float(max(1, w)), body_h / float(max(1, h)))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img2 = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (panel_w - nw) // 2
    y0 = head_h + (body_h - nh) // 2
    panel[y0:y0 + nh, x0:x0 + nw] = img2

    cv2.rectangle(panel, (0, 0), (panel_w - 1, head_h), (235, 235, 235), -1)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (180, 180, 180), 1)
    cv2.putText(panel, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    return panel


def hand_crop_panel(frame, bbox, out_size):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    side = max(32, int(max(bw, bh) * 1.8))

    xx1 = max(0, cx - side // 2)
    yy1 = max(0, cy - side // 2)
    xx2 = min(w, cx + side // 2)
    yy2 = min(h, cy + side // 2)

    crop = frame[yy1:yy2, xx1:xx2]
    if crop.size == 0:
        crop = np.full((h, w, 3), 230, dtype=np.uint8)
    crop = cv2.resize(crop, out_size, interpolation=cv2.INTER_CUBIC)
    return crop


def choose_critical_frames(valid_mask, confidence, num_frames):
    if num_frames <= 0: return [0, 0, 0, 0]

    if len(valid_mask) != num_frames: valid_mask = np.zeros((num_frames,), dtype=bool)
    if len(confidence) != num_frames: confidence = np.zeros((num_frames,), dtype=np.float32)

    idx = np.arange(num_frames)
    idx_valid = idx[valid_mask]
    idx_invalid = idx[~valid_mask]

    f_invalid = int(idx_invalid[len(idx_invalid) // 2]) if len(idx_invalid) > 0 else int(num_frames // 2)

    if len(idx_valid) > 0:
        c = confidence[idx_valid]
        order = np.argsort(c)
        f_worst_valid = int(idx_valid[order[0]])
        f_median_valid = int(idx_valid[order[len(order) // 2]])
        f_best_valid = int(idx_valid[order[-1]])
    else:
        f_worst_valid = f_median_valid = f_best_valid = f_invalid

    return [f_invalid, f_worst_valid, f_median_valid, f_best_valid]


def build_timeline_panel(valid_mask, confidence, panel_w=1100, panel_h=400, markers=None):
    panel = np.full((panel_h, panel_w, 3), 248, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (180, 180, 180), 1)

    t = len(valid_mask)
    if t == 0: return panel

    if len(confidence) != t: confidence = np.zeros((t,), dtype=np.float32)

    left, right, top, bottom = 40, panel_w - 20, 30, panel_h - 30
    width, height = max(1, right - left), max(1, bottom - top)
    xs = left + (np.arange(t) * width / max(1, t - 1)).astype(np.int32)

    for i, x in enumerate(xs):
        color = (90, 190, 90) if bool(valid_mask[i]) else (90, 90, 220)
        cv2.line(panel, (int(x), top), (int(x), top + 26), color, 1, cv2.LINE_AA)

    conf = np.clip(confidence, 0.0, 1.0)
    ys = (bottom - (conf * (height - 36))).astype(np.int32)
    for i in range(1, t):
        cv2.line(panel, (int(xs[i - 1]), int(ys[i - 1])), (int(xs[i]), int(ys[i])), (255, 120, 0), 2, cv2.LINE_AA)

    marker_colors = [(30, 30, 30), (0, 100, 220), (70, 70, 70), (0, 160, 0)]
    for mi, m in enumerate(markers or []):
        x = int(left + (int(np.clip(m, 0, t - 1)) * width / max(1, t - 1)))
        cv2.line(panel, (x, top), (x, bottom), marker_colors[mi % len(marker_colors)], 2, cv2.LINE_AA)

    return panel


def make_contact_sheet(images, thumb_width=700):
    if not images: return None
    resized = [cv2.resize(im, (thumb_width, max(1, int(im.shape[0] * thumb_width / max(1, im.shape[1])))), interpolation=cv2.INTER_AREA) for im in images]
    n, cols = len(resized), int(np.ceil(np.sqrt(len(resized))))
    rows, cell_h = int(np.ceil(n / cols)), max(im.shape[0] for im in resized)

    sheet = np.full((rows * cell_h, cols * thumb_width, 3), 248, dtype=np.uint8)
    for idx, im in enumerate(resized):
        r, c = divmod(idx, cols)
        y0, x0 = r * cell_h, c * thumb_width
        sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return sheet


def compute_metrics(data, video_w=None, video_h=None):
    keypoints = np.asarray(data.get('keypoints', []), dtype=np.float32)
    valid_mask = np.asarray(data.get('valid_mask', []), dtype=bool)
    confidence = np.asarray(data.get('confidence', []), dtype=np.float32)
    bboxes = np.asarray(data.get('bboxes', []), dtype=np.float32)

    if keypoints.ndim != 3 or keypoints.shape[1:] != (21, 2): return None

    t = keypoints.shape[0]
    if valid_mask.shape[0] != t: valid_mask = np.zeros((t,), dtype=bool)
    if confidence.shape[0] != t: confidence = np.zeros((t,), dtype=np.float32)

    valid_ratio = float(valid_mask.mean()) if t > 0 else 0.0
    conf_mean = float(confidence.mean()) if t > 0 else 0.0

    wrist = keypoints[:, 0, :] if t > 0 else np.zeros((0, 2), dtype=np.float32)
    wrist_motion_mean = float(np.linalg.norm(wrist[1:] - wrist[:-1], axis=1).mean()) if len(wrist) > 1 else 0.0

    oob_ratio = 0.0
    if video_w and video_h and t > 0:
        x, y = keypoints[:, :, 0], keypoints[:, :, 1]
        oob_ratio = float(((x < 0) | (x >= video_w) | (y < 0) | (y >= video_h)).mean())

    bbox_area_ratio_mean = 0.0
    if bboxes.ndim == 2 and bboxes.shape[0] == t and bboxes.shape[1] == 4 and video_w and video_h:
        area = np.clip(bboxes[:, 2] - bboxes[:, 0], 0, None) * np.clip(bboxes[:, 3] - bboxes[:, 1], 0, None)
        bbox_area_ratio_mean = float((area / max(1, video_w * video_h)).mean())

    return {
        'num_frames': int(t), 'num_valid': int(valid_mask.sum()), 'valid_ratio': valid_ratio,
        'conf_mean': conf_mean, 'wrist_motion_mean': wrist_motion_mean,
        'oob_ratio': oob_ratio, 'bbox_area_ratio_mean': bbox_area_ratio_mean,
    }


def quality_label(m, args):
    if m['valid_ratio'] < args.thr_valid_bad or m['oob_ratio'] > args.thr_oob_bad: return 'bad'
    if m['valid_ratio'] >= args.thr_valid_good and m['wrist_motion_mean'] >= args.thr_motion_low: return 'good'
    return 'warn'


def render_sample_image(record, out_path):
    data, m = record['data'], record['metrics']
    keypoints = np.asarray(data.get('keypoints', []), dtype=np.float32)
    valid_mask = np.asarray(data.get('valid_mask', []), dtype=bool)
    confidence = np.asarray(data.get('confidence', []), dtype=np.float32)
    bboxes = np.asarray(data.get('bboxes', []), dtype=np.float32)
    t = len(keypoints)

    if t == 0: return None
    if valid_mask.shape[0] != t: valid_mask = np.zeros((t,), dtype=bool)
    if confidence.shape[0] != t: confidence = np.zeros((t,), dtype=np.float32)
    if bboxes.ndim != 2 or bboxes.shape[0] != t or bboxes.shape[1] != 4: bboxes = np.tile([[0, 0, 1, 1]], (t, 1))

    local_indices = choose_critical_frames(valid_mask, confidence, t)
    labels = ['invalid-mid', 'worst-valid', 'median-valid', 'best-valid']

    overlays = []
    for i, local_idx in enumerate(local_indices):
        frame, _, _ = read_frame(record['video_path'], int(local_idx))
        if frame is None: return None
        bbox = bboxes[int(np.clip(local_idx, 0, len(bboxes) - 1))]
        overlays.append((draw_overlay(frame, keypoints[int(local_idx)], bbox), bbox, f'{labels[i]} | t={local_idx}', local_idx))

    top_row = np.concatenate([make_panel(ov, title, 500, 340) for ov, _, title, _ in overlays], axis=1)
    
    crop = hand_crop_panel(overlays[1][0], overlays[1][1], (900, 360))
    crop_panel = draw_panel_title(crop, f'Hand Crop (worst-valid) t={overlays[1][3]}')
    timeline = draw_panel_title(build_timeline_panel(valid_mask, confidence, 1100, 400, local_indices), 'Temporal Diagnostics')

    h_target = max(crop_panel.shape[0], timeline.shape[0])
    crop_panel = cv2.resize(crop_panel, (crop_panel.shape[1], h_target))
    timeline = cv2.resize(timeline, (timeline.shape[1], h_target))
    bottom_row = np.concatenate([crop_panel, timeline], axis=1)

    width = max(top_row.shape[1], bottom_row.shape[1])
    top_row = np.concatenate([top_row, np.full((top_row.shape[0], width - top_row.shape[1], 3), 248, dtype=np.uint8)], axis=1) if top_row.shape[1] != width else top_row
    bottom_row = np.concatenate([bottom_row, np.full((bottom_row.shape[0], width - bottom_row.shape[1], 3), 248, dtype=np.uint8)], axis=1) if bottom_row.shape[1] != width else bottom_row

    canvas = np.concatenate([top_row, bottom_row], axis=0)
    cv2.putText(canvas, f"{record['file'].name} | label={record['label']} | valid={m['valid_ratio']:.3f} conf={m['conf_mean']:.3f} oob={m['oob_ratio']:.3f}", 
                (12, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    return canvas


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    keypoints_dir = Path(args.keypoints_dir)
    output_dir = Path(args.output_dir)
    base_dir = Path(args.base_dir).resolve()

    overlays_dir = output_dir / 'overlays'
    sheets_dir = output_dir / 'contact_sheets'
    for d in [output_dir, overlays_dir, sheets_dir]: d.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(keypoints_dir.glob('*.pkl'))
    if not pkl_files:
        print(f'No PKL files found in {keypoints_dir}')
        return

    records, skipped = [], 0
    print("Loading extracted keypoints for unified sanity check...")
    for p in pkl_files:
        with open(p, 'rb') as f: data = pickle.load(f)

        video_path = resolve_video_path(data, base_dir)
        if video_path is None:
            skipped += 1; continue

        _, vw, vh = read_frame(video_path, 0)
        if vw is None or vh is None:
            skipped += 1; continue

        metrics = compute_metrics(data, vw, vh)
        if metrics is None:
            skipped += 1; continue

        records.append({'file': p, 'video_path': video_path, 'data': data, 'metrics': metrics, 'label': quality_label(metrics, args)})

    if not records:
        print('No valid records to analyze.')
        return

    csv_path = output_dir / 'per_file_metrics.csv'
    fieldnames = ['file', 'video_path', 'subject', 'scene', 'video_id', 'label', 'num_frames', 'num_valid', 'valid_ratio', 'conf_mean', 'wrist_motion_mean', 'oob_ratio', 'bbox_area_ratio_mean']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            d, m = r['data'], r['metrics']
            writer.writerow({
                'file': r['file'].name, 'video_path': str(r['video_path']),
                'subject': d.get('subject', ''), 'scene': d.get('scene', ''), 'video_id': d.get('video_id', d.get('video_num', '')),
                'label': r['label'], **m
            })

    counts = {lbl: sum(1 for r in records if r['label'] == lbl) for lbl in ['good', 'warn', 'bad']}
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump({
            'num_files_scanned': len(pkl_files), 'num_files_analyzed': len(records), 'num_files_skipped': skipped,
            'quality_counts': counts,
            'valid_ratio': {'mean': float(np.mean([r['metrics']['valid_ratio'] for r in records]))}
        }, f, indent=2)

    by_valid = sorted(records, key=lambda r: r['metrics']['valid_ratio'])
    groups = [
        ('worst', by_valid[:min(args.visuals_per_group, len(by_valid))]), 
        ('best', list(reversed(by_valid[-min(args.visuals_per_group, len(by_valid)):]))), 
        ('random', rng.sample(records, min(args.visuals_per_group, len(records))))
    ]

    print("Rendering contact sheets and overlays...")
    for name, group in groups:
        rendered = [render_sample_image(r, overlays_dir / name / f'{i:03d}_{r["file"].stem}.png') for i, r in enumerate(group, 1)]
        rendered = [v for v in rendered if v is not None]
        if rendered: cv2.imwrite(str(sheets_dir / f'{name}_contact_sheet.png'), make_contact_sheet(rendered, args.thumb_width))

    print(f'\nSanity check complete. Analyzed: {len(records)}/{len(pkl_files)} | Quality: {counts}')


if __name__ == '__main__':
    main()