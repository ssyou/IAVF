### Dataset & Features
1. Datasets：The proposed AVEPR and AVELPR datasets can be download from Google Drive [link](https://drive.google.com/drive/folders/1lDUiDtJhTc-hzK-AaEAuAUZ2VCy4e73F?usp=sharing)
2. Features <br>
  extract frame-level image from videos: python extract_frames.py <br>
  extract 2D frame-level features: python extract_3D_feat.py <br>
  extract 3D snippet-level features: python extract_rgb_feat.py <br>
  extract audio samples from videos: python extract_audio.py <br>
  extract audio features: python wave audio_feature_extractor.py <br>
  (You can also extract audio features with [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn))
  
 ### Requirements
```bash
conda env create name environment_name -f IAVF.yml
```

