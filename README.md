# Incremental Audio-Visual Fusion for Person Recognition in Earthquake Scene
<div align="center">
  <img src="https://github.com/ssyou/IAVF/blob/main/figs/framework.png">
</div>

## Dataset & Features
1. Datasets：The proposed AVEPR and AVELPR datasets can be download from Google Drive [link](https://drive.google.com/drive/folders/1lDUiDtJhTc-hzK-AaEAuAUZ2VCy4e73F?usp=sharing)
2. Features <br>
  extract frame-level image from videos: 
  ```bash
  python extract_frames.py
  ```
  extract 2D frame-level features: 
  ```bash
  python extract_3D_feat.py
  ```
  extract 3D snippet-level features: 
  ```bash
  python extract_rgb_feat.py
  ```
  extract audio samples from videos: 
  ```bash
  python extract_audio.py
  ```
  extract audio features: 
  ```bash
  python wave audio_feature_extractor.py
  ```
  (You can also extract audio features with [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn))
  
## Requirements
```bash
conda env create -f IAVF.yml
```

## Incremental Audio-Visual Fusion 
Training
```bash
python main.py --gpu 0 --K 1 --dataset AVE --mode train --batch-size 32 --epochs 10 
```

Testing
```bash
python main.py --gpu 0 --K 1 --dataset AVE --mode test --batch-size 32 --epochs 10 
```
