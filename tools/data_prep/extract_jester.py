#!/usr/bin/env python3
"""Extract 2D keypoints from Jester dataset using MediaPipe."""

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
    parser = argparse.ArgumentParser(description="Extract Jester keypoints with MediaPipe")
    parser.add_argument('--jester-root', type=str, required=True,
                       help="Path to Jester/20bn-jester-v1")
    parser.add_argument('--output-dir', type=str, required=True,
                       help="Output directory for PKL keypoints")
    parser.add_argument('--max-videos', type=int, default=None,
                       help="Max number of videos to process (for debugging)")
    parser.add_argument('--gpu', type=int, default=0,
                       help="Argomento mantenuto per compatibilità, MediaPipe userà la CPU")
    parser.add_argument('--det-threshold', type=float, default=0.3,
                       help="Detection confidence threshold")
    parser.add_argument('--pose-threshold', type=float, default=0.3,
                       help="Tracking confidence threshold")
    parser.add_argument('--min-valid-ratio', type=float, default=0.3,
                       help="Minimum ratio of valid frames (default: 0.3)")
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
    T = len(keypoints_array)
    
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 2:
        return keypoints_array
    
    for joint in range(21):
        for coord in range(2):
            valid_values = keypoints_array[valid_indices, joint, coord]
            f = interp1d(
                valid_indices, 
                valid_values,
                kind='linear',
                bounds_error=False,
                fill_value=(valid_values[0], valid_values[-1])
            )
            keypoints_array[:, joint, coord] = f(np.arange(T))
    
    return keypoints_array


def smooth_keypoints(keypoints_array):
    """Temporal smoothing with Savitzky-Golay filter."""
    T = keypoints_array.shape[0]
    if T <= 3:
        return keypoints_array
    
    smoothed = keypoints_array.copy()
    wl = min(5, T)
    if wl % 2 == 0:
        wl -= 1
    
    for joint in range(21):
        for coord in range(2):
            smoothed[:, joint, coord] = savgol_filter(
                keypoints_array[:, joint, coord],
                window_length=wl,
                polyorder=2,
                mode='nearest'
            )
    
    return smoothed


def process_video(video_dir, detector, args):
    """Process a single Jester video (folder of frames) using MediaPipe."""
    frame_files = sorted(video_dir.glob('*.jpg'))
    
    if len(frame_files) == 0:
        return None, "empty_folder"
    
    keypoints_list = []
    confidence_list = []
    bbox_list = []
    valid_mask = []
    
    last_valid_kps = None
    last_valid_bbox = None
    last_valid_conf = 0.0
    
    # Read the first frame just to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        return None, "cannot_read_frames"
    h, w, _ = first_frame.shape

    # Reset detector state for each new video sequence
    detector.reset()

    for frame_idx, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detector.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            # No detection: use last valid or zeros
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
            # Valid detection (take the first hand detected)
            hand_landmarks = results.multi_hand_landmarks[0]
            
            kps_px = np.zeros((21, 2), dtype=np.float32)
            for i, lm in enumerate(hand_landmarks.landmark):
                kps_px[i, 0] = lm.x * w
                kps_px[i, 1] = lm.y * h
                
            # Synthetic Bounding Box generation from keypoints
            x_min, x_max = kps_px[:, 0].min(), kps_px[:, 0].max()
            y_min, y_max = kps_px[:, 1].min(), kps_px[:, 1].max()
            pad_x, pad_y = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
            bbox = [
                max(0, x_min - pad_x), max(0, y_min - pad_y),
                min(w, x_max + pad_x), min(h, y_max + pad_y)
            ]
            
            conf = 1.0  # MediaPipe returns coords, we assume 1.0 for valid frames
            
            keypoints_list.append(kps_px.copy())
            bbox_list.append(bbox)
            confidence_list.append(conf)
            valid_mask.append(True)
            
            # Update memory for forward-fill
            last_valid_kps = kps_px
            last_valid_bbox = bbox
            last_valid_conf = conf
    
    valid_mask = np.array(valid_mask, dtype=bool)
    valid_count = valid_mask.sum()
    
    if valid_count < len(frame_files) * args.min_valid_ratio:
        return None, "low_valid_ratio"
    
    if not args.no_interpolate and valid_count >= 2:
        keypoints_array = interpolate_keypoints(keypoints_list, valid_mask)
    else:
        keypoints_array = np.array(keypoints_list, dtype=np.float32)
    
    if not args.no_smooth:
        keypoints_array = smooth_keypoints(keypoints_array)
    
    result = {
        'video_id': video_dir.name,
        'keypoints': keypoints_array,          
        'confidence': confidence_list,         
        'bboxes': bbox_list,                   
        'valid_mask': valid_mask,              
        'num_frames': len(frame_files),
        'num_valid': int(valid_count),
        'interpolated': not args.no_interpolate,
        'smoothed': not args.no_smooth
    }
    
    return result, None


def main():
    args = parse_args()
    
    jester_root = Path(args.jester_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not jester_root.exists():
        print(f"ERROR: Jester root not found: {jester_root}")
        return
    
    video_dirs = sorted([d for d in jester_root.iterdir() if d.is_dir()])
    
    if args.max_videos:
        video_dirs = video_dirs[:args.max_videos]
    
    print(f"\n{'='*60}")
    print(f"JESTER KEYPOINT EXTRACTION (MEDIAPIPE)")
    print(f"{'='*60}")
    print(f"Input: {jester_root}")
    print(f"Output: {output_dir}")
    print(f"Total videos: {len(video_dirs)}")
    print(f"Det threshold: {args.det_threshold}")
    print(f"Tracking threshold: {args.pose_threshold}")
    print(f"Min valid ratio: {args.min_valid_ratio}")
    print(f"Interpolation: {'OFF' if args.no_interpolate else 'ON'}")
    print(f"Smoothing: {'OFF' if args.no_smooth else 'ON (Savitzky-Golay)'}")
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
    total_frames = 0
    failure_counts = {}
    
    pbar = tqdm(video_dirs, desc="MediaPipe Extraction", mininterval=2.0, ncols=100)
    for video_dir in pbar:
        output_file = output_dir / f"{video_dir.name}.pkl"
        
        if args.resume and output_file.exists():
            skipped_count += 1
            continue
        
        result, failure_reason = process_video(video_dir, detector, args)
        
        if result is None:
            failed_count += 1
            failure_counts[failure_reason] = failure_counts.get(failure_reason, 0) + 1
            continue
        
        with open(output_file, 'wb') as f:
            pickle.dump(result, f)
        
        success_count += 1
        total_valid_frames += result['num_valid']
        total_frames += result['num_frames']
        
        if success_count % 100 == 0:
            avg_valid_ratio = total_valid_frames / total_frames if total_frames > 0 else 0.0
            pbar.set_postfix({
                'success': success_count,
                'failed': failed_count,
                'avg_valid': f"{avg_valid_ratio*100:.1f}%"
            })
    
    pbar.close()
    detector.close()
    
    avg_valid_ratio = total_valid_frames / total_frames if total_frames > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Success: {success_count}/{len(video_dirs)}")
    print(f"Failed: {failed_count}/{len(video_dirs)}")
    if args.resume:
        print(f"Skipped (already processed): {skipped_count}/{len(video_dirs)}")
    if failure_counts:
        print(f"Failure reasons:")
        for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")
    print(f"Average valid ratio: {avg_valid_ratio*100:.1f}%")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()