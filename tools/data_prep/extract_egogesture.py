#!/usr/bin/env python3
"""Extract 2D keypoints from EgoGesture RGB videos using MediaPipe."""

import sys
import argparse
import cv2
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import mediapipe as mp

def parse_args():
    parser = argparse.ArgumentParser(description="Extract EgoGesture keypoints with MediaPipe")
    parser.add_argument('--ego-root', type=str, required=True,
                       help="Path to dataset/EgoGesture")
    parser.add_argument('--output-dir', type=str, required=True,
                       help="Output directory for PKL keypoints")
    parser.add_argument('--max-videos', type=int, default=None,
                       help="Max number of videos to process (for debugging)")
    parser.add_argument('--gpu', type=int, default=0,
                       help="Kept for compatibility; MediaPipe will run on CPU")
    parser.add_argument('--det-threshold', type=float, default=0.3,
                       help="Detection confidence threshold")
    parser.add_argument('--pose-threshold', type=float, default=0.3,
                       help="Tracking confidence threshold")
    parser.add_argument('--min-valid-ratio', type=float, default=0.05,
                       help="Minimum ratio of valid processed frames")
    parser.add_argument('--min-valid-frames', type=int, default=2,
                       help="Minimum number of valid processed frames")
    parser.add_argument('--no-interpolate', action='store_true',
                       help="Disable linear interpolation (use forward-fill)")
    parser.add_argument('--no-smooth', action='store_true',
                       help="Disable temporal smoothing")
    parser.add_argument('--resume', action='store_true',
                       help="Skip already processed videos")
    return parser.parse_args()


def interpolate_keypoints(keypoints_list, valid_mask):
    """Linear interpolation for missing keypoints."""
    keypoints_array = np.array(keypoints_list, dtype=np.float32)
    num_frames = len(keypoints_array)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return keypoints_array

    for joint in range(21):
        for coord in range(2):
            valid_values = keypoints_array[valid_indices, joint, coord]
            interp_fn = interp1d(
                valid_indices,
                valid_values,
                kind='linear',
                bounds_error=False,
                fill_value=(valid_values[0], valid_values[-1])
            )
            keypoints_array[:, joint, coord] = interp_fn(np.arange(num_frames))

    return keypoints_array


def smooth_keypoints(keypoints_array):
    """Temporal smoothing with Savitzky-Golay filter."""
    num_frames = keypoints_array.shape[0]
    if num_frames <= 3:
        return keypoints_array

    smoothed = keypoints_array.copy()
    window_length = min(5, num_frames)
    if window_length % 2 == 0:
        window_length -= 1

    for joint in range(21):
        for coord in range(2):
            smoothed[:, joint, coord] = savgol_filter(
                keypoints_array[:, joint, coord],
                window_length=window_length,
                polyorder=2,
                mode='nearest'
            )

    return smoothed


def discover_videos(ego_root):
    """Robust search for all EgoGesture videos and their labels."""
    records = []
    labels_root = ego_root / 'labels-final-revised1'

    print(f"\nSearching root: {ego_root}")
    if not labels_root.exists():
        print(f"CRITICAL ERROR: Label folder not found in {labels_root}")
        return records

    found_videos = 0
    for v_folder in ego_root.glob('videos_*'):
        for video_path in v_folder.rglob('*'):
            if video_path.suffix.lower() not in ['.avi', '.mp4']:
                continue
            
            found_videos += 1
            parts = video_path.parts
            subject_id = None
            scene_id = None
            
            for p in parts:
                p_lower = p.lower()
                if p_lower.startswith('subject'):
                    subject_id = p_lower.replace('subject', '')
                elif p_lower.startswith('scene'):
                    scene_id = p
            
            if not subject_id or not scene_id:
                continue
                
            video_num = video_path.stem.lower().replace('rgb', '')
            csv_path = labels_root / f"subject{subject_id}" / scene_id / f"Group{video_num}.csv"
            
            records.append({
                'subject': subject_id,
                'scene': scene_id,
                'video_num': video_num,
                'video_path': video_path,
                'csv_path': csv_path
            })
            
    print(f"Found {found_videos} video files (.avi/.mp4) on disk.")
    print(f"Matched to Subject/Scene: {len(records)} valid videos.")
    return records


def process_video(video_record, detector, args):
    """Process a single EgoGesture continuous RGB video using MediaPipe."""
    video_path = video_record['video_path']
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, "cannot_open_video"

    detector.reset()

    keypoints_list = []
    confidence_list = []
    bbox_list = []
    valid_mask = []

    last_valid_kps = None
    last_valid_bbox = None
    last_valid_conf = 0.0

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None, "empty_video"
    h, w, _ = frame.shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(frame_rgb)

        if not results.multi_hand_landmarks:
            if last_valid_kps is not None:
                keypoints_list.append(last_valid_kps.copy())
                bbox_list.append(last_valid_bbox)
                confidence_list.append(last_valid_conf)
            else:
                keypoints_list.append(np.zeros((21, 2), dtype=np.float32))
                bbox_list.append([0, 0, 0, 0])
                confidence_list.append(0.0)
            valid_mask.append(False)
        else:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            kps_px = np.zeros((21, 2), dtype=np.float32)
            for i, lm in enumerate(hand_landmarks.landmark):
                kps_px[i, 0] = lm.x * w
                kps_px[i, 1] = lm.y * h

            x_min, x_max = kps_px[:, 0].min(), kps_px[:, 0].max()
            y_min, y_max = kps_px[:, 1].min(), kps_px[:, 1].max()
            pad_x, pad_y = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
            bbox = [
                max(0, x_min - pad_x), max(0, y_min - pad_y),
                min(w, x_max + pad_x), min(h, y_max + pad_y)
            ]

            conf = 1.0

            keypoints_list.append(kps_px.copy())
            bbox_list.append(bbox)
            confidence_list.append(conf)
            valid_mask.append(True)

            last_valid_kps = kps_px
            last_valid_bbox = bbox
            last_valid_conf = conf

    cap.release()

    if not keypoints_list:
        return None, "no_frames_extracted"

    valid_mask = np.array(valid_mask, dtype=bool)
    valid_count = int(valid_mask.sum())
    processed_frames = len(keypoints_list)

    if valid_count < args.min_valid_frames:
        return None, f"too_few_valid_frames({valid_count}/{processed_frames})"
    if valid_count < processed_frames * args.min_valid_ratio:
        return None, f"low_valid_ratio({valid_count}/{processed_frames}={100*valid_count/processed_frames:.1f}%)"

    if not args.no_interpolate and valid_count >= 2:
        keypoints_array = interpolate_keypoints(keypoints_list, valid_mask)
    else:
        keypoints_array = np.array(keypoints_list, dtype=np.float32)

    if not args.no_smooth:
        keypoints_array = smooth_keypoints(keypoints_array)

    sample_id = f"Subject{video_record['subject']}_{video_record['scene']}_rgb{video_record['video_num']}"

    return {
        'video_id': sample_id,
        'subject': video_record['subject'],
        'scene': video_record['scene'],
        'video_num': video_record['video_num'],
        'video_path': str(video_path),
        'csv_path': str(video_record['csv_path']) if video_record['csv_path'].exists() else None,
        'keypoints': keypoints_array,
        'confidence': confidence_list,
        'bboxes': bbox_list,
        'valid_mask': valid_mask,
        'num_frames': processed_frames,
        'num_valid': valid_count,
        'interpolated': not args.no_interpolate,
        'smoothed': not args.no_smooth,
    }, None


def main():
    args = parse_args()

    ego_root = Path(args.ego_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ego_root.exists():
        print(f"ERROR: EgoGesture root not found: {ego_root}")
        return

    video_records = discover_videos(ego_root)
    if args.max_videos:
        video_records = video_records[:args.max_videos]

    print(f"\n{'='*60}")
    print("EGOGESTURE KEYPOINT EXTRACTION (MEDIAPIPE)")
    print(f"{'='*60}")
    print(f"Input: {ego_root}")
    print(f"Output: {output_dir}")
    print(f"Total Videos discovered: {len(video_records)}")
    print(f"Det threshold: {args.det_threshold}")
    print(f"Pose (Tracking) threshold: {args.pose_threshold}")
    print(f"Min valid ratio: {args.min_valid_ratio}")
    print(f"{'='*60}\n")

    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.det_threshold,
        min_tracking_confidence=args.pose_threshold
    )

    success_count = 0
    failed_count = 0
    skipped_count = 0
    total_valid_frames = 0
    total_processed_frames = 0
    failure_counts = {}

    pbar = tqdm(video_records, desc="MediaPipe Extraction", mininterval=2.0, ncols=100)
    for video_record in pbar:
        out_filename = f"Subject{video_record['subject']}_{video_record['scene']}_rgb{video_record['video_num']}.pkl"
        output_file = output_dir / out_filename

        if args.resume and output_file.exists():
            skipped_count += 1
            continue

        result, failure_reason = process_video(video_record, detector, args)
        if result is None:
            failed_count += 1
            failure_counts[failure_reason] = failure_counts.get(failure_reason, 0) + 1
            continue

        with open(output_file, 'wb') as handle:
            pickle.dump(result, handle)

        success_count += 1
        total_valid_frames += result['num_valid']
        total_processed_frames += result['num_frames']

        if success_count % 50 == 0:
            avg_valid_ratio = total_valid_frames / total_processed_frames if total_processed_frames > 0 else 0.0
            pbar.set_postfix({
                'success': success_count,
                'failed': failed_count,
                'avg_v': f"{avg_valid_ratio*100:.1f}%"
            })

    pbar.close()
    detector.close()

    avg_valid_ratio = total_valid_frames / total_processed_frames if total_processed_frames > 0 else 0.0
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Success: {success_count}/{len(video_records)}")
    print(f"Failed: {failed_count}/{len(video_records)}")
    if args.resume:
        print(f"Skipped (already processed): {skipped_count}/{len(video_records)}")
    if failure_counts:
        print(f"Failure reasons:")
        for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")
    print(f"Average valid ratio: {avg_valid_ratio*100:.1f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()