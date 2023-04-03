# Extract audio waveforms from videos. The extracted audios will be in the data/LLP_dataset/audio folder. moviepy library is used to read videos and extract audios.

import os
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip
import pdb
video_pth =  "/data/yss/dataset/AVE_Dataset/AVE"
sound_list = os.listdir(video_pth)
save_pth =  "/data/yss/dataset/AVE_Dataset/AVE/compress_0330/audio_train_256"
if not os.path.exists(save_pth):
    os.makedirs(save_pth)
for video in sound_list:
    # pdb.set_trace()
    video_name = os.path.join(video_pth, video)
    # name_vid_list = os.listdir(name)
    # for video in name_vid_list:
    # name_video = os.path.join(name, video)
    audio_name = video[:-4] + '.wav'
    exist_lis = os.listdir(save_pth)
    if audio_name in exist_lis:
        print("already exist!")
        continue
    try:
        video = VideoFileClip(video_name)
        audio = video.audio
        audio.write_audiofile(os.path.join(save_pth, audio_name), fps=16000)
        print("finish video id: " + audio_name)
    except:
        print("cannot load ", video_name)
#所有视频放一个文件夹里
# video_pth =  "data/LLP_dataset/video"
# sound_list = os.listdir(video_pth)
# save_pth =  "data/LLP_dataset/audio"

# for audio_id in sound_list:
#     name = os.path.join(video_pth, audio_id)
#     audio_name = audio_id[:-4] + '.wav'
#     exist_lis = os.listdir(save_pth)
#     if audio_name in exist_lis:
#         print("already exist!")
#         continue
#     try:
#         video = VideoFileClip(name)
#         audio = video.audio
#         audio.write_audiofile(os.path.join(save_pth, audio_name), fps=16000)
#         print("finish video id: " + audio_name)
#     except:
#         print("cannot load ", name)