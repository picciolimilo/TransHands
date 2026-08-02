# Datasets

First, create the base directory for datasets:
```bash
mkdir -p dataset
```

## 2D-to-3D Lifting / Pretraining

### Re:InterHand Dataset
Download the annotation data from the [official website](https://mks0601.github.io/ReInterHand/).

The download file needed are:
- `download_checksum_framelist.py`
- `download_origin_fits.py` -> The `origin_fits/left/Keypoints/` and `origin_fits/right/Keypoints/` sections.

```bash
 mkdir -p dataset/ReInterHand
 # Ensure 'origin_fits' folder is inside dataset/ReInterHand/
```

### AssemblyHands Dataset
Download the annotation data from the [official website](https://assemblyhands.github.io).

```bash
 mkdir -p dataset/AssemblyHands
 # Insert the .json annotation files here
```

### GigaHands Dataset
Download the data from the [official website](https://ivl.cs.brown.edu/research/gigahands.html).

- [Hand Poses Set](https://g-ad09a0.56197.5898.data.globus.org/hand_poses.tar.gz)

```bash
 mkdir -p dataset/GigaHands
 # Extract the folder here
```

## Gesture Recognition

### Jester Dataset
The full dataset can be downloaded from its [official kaggle page](https://www.kaggle.com/datasets/sanjanatg26/20bn-jester-v1-complete).

```bash
 mkdir -p dataset/Jester/
 # Extract the downloaded archive here
```

The main video archive does not contain the class labels. Download the official CSV annotation files directly into your dataset folder:

```bash
cd dataset/Jester
mkdir -p annotations
cd annotations

wget https://raw.githubusercontent.com/udacity/CVND---Gesture-Recognition/master/20bn-jester-v1/annotations/jester-v1-labels.csv
wget https://raw.githubusercontent.com/udacity/CVND---Gesture-Recognition/master/20bn-jester-v1/annotations/jester-v1-train.csv
wget https://raw.githubusercontent.com/udacity/CVND---Gesture-Recognition/master/20bn-jester-v1/annotations/jester-v1-validation.csv
```

TransHands requires 2D keypoints as input. 
Run the keypoint extraction script based on MediaPipe:
```bash
# From the project root
python tools/data_prep/extract_jester.py \
    --jester-root dataset/Jester/20bn-jester-v1 \
    --output-dir dataset/Jester/keypoints_mediapipe
```
### EgoGesture Dataset
The dataset can be obtained from the [official website](https://nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html) (requires access request).

```bash
 mkdir -p dataset/EgoGesture/
 # Extract the downloaded archive here
```

TransHands requires 2D keypoints as input. 
Run the keypoint extraction script based on MediaPipe:
```bash
# Make sure you are in the root directory of the TransHands project
python tools/data_prep/extract_egogesture.py \
    --ego-root dataset/EgoGesture \
    --output-dir dataset/EgoGesture/keypoints_mediapipe
```
