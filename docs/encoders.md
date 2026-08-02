# Setup Encoder & Pretrained Weights

We use several state-of-the-art backbones located in the `external/` directory. Follow the steps below to set up the desired encoder.

First, create the base directory for external repositories:
```bash
mkdir -p external
```

## MotionBERT
1. Clone MotionBERT:
    ```bash
    cd external
    git clone https://github.com/Walter0807/MotionBERT.git
    ```

2. Download MotionBERT [pretrained weights for 3D pose estimation](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBdkFkaDBMU2pFT2xnU29UcXR5UjVac2dpOF9aP2U9cm40Vkpm&id=A5438CD242871DF0%21170&cid=A5438CD242871DF0).

3. Place the file `best_epoch.bin` in:
    ```bash
    cd MotionBERT
    mkdir -p checkpoint/pretrain
    # Move best_epoch.bin to external/MotionBERT/checkpoint/pretrain/
    ```

## MixSTE
1. Clone MixSTE:
    ```bash
    cd external
    git clone https://github.com/JinluZhang1126/MixSTE.git
    ```

2. Download MixSTE [pretrained weights](https://drive.google.com/drive/folders/1G2mlMHebM6KcbI45FszlosIHgA4jiR3Y).  
    *Recommended*: `best_epoch_cpn_243f.bin` (Seq Len 243).

3. Place the file in:
    ```bash
    cd MixSTE
    mkdir -p checkpoint/pretrain
    # Move best_epoch_cpn_243f.bin to external/MixSTE/checkpoint/pretrain/
    ```

## PoseFormerV2
1. Clone PoseFormerV2:
    ```bash
    cd external
    git clone https://github.com/QitaoZhao/PoseFormerV2.git
    ```
2. Download PoseFormerV2 [pretrained weights](https://drive.google.com/file/d/14SpqPyq9yiblCzTH5CorymKCUsXapmkg/view).  
    *Recommended*: The 243 Frames version with best MPJPE.

3. Place the file `27_243_45.2.bin` in:
    ```bash
    cd PoseFormerV2
    mkdir -p checkpoint/pretrain
    # Move 27_243_45.2.bin to external/PoseFormerV2/checkpoint/pretrain/
    ```

## ST-GCN
1. Clone ST-GCN:
    ```bash
    cd external
    git clone https://github.com/yysijie/st-gcn.git
    ```

2. Download ST-GCN [pretrained weights](https://drive.google.com/drive/folders/1IYKoSrjeI3yYJ9bO0_z_eDo92i7ob_aF). 
    *Recommended*: The Kinetic version.

3. Place the `st_gcn.kinetics.pt` file in:
    ```bash
    cd st-gcn
    mkdir -p checkpoint/pretrain
    # Move st_gcn.kinetics.pt to external/st-gcn/checkpoint/pretrain/
    ```

