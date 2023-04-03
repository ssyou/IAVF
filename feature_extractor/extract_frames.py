# Extract video frames from videos. The extracted frames will be in the data/LLP_dataset/frame folder.
import shutil
import subprocess
import os
import argparse
import glob
import pdb
#对于val_256的抽取有问题 有的是mkv后缀 有的是mp4后缀
def extract_frames(video, dst):
    command1 = 'ffmpeg '
    command1 += '-i ' + video + " "
    command1 += '-y' + " "
    command1 += "-r " + "8 "
    command1 += '{0}/%06d.jpg'.format(dst)
    print(command1)
    #    print command1
    os.system(command1)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='/data/yss/dataset/AVE_Dataset/AVE/compress_0330/frame_test_256')
    parser.add_argument('--video_path', dest='video_path', type=str, default='/data/yss/dataset/AVE_Dataset/AVE')
    args = parser.parse_args()

    vid_list = os.listdir(args.video_path)

    for video in vid_list:
        name_video = os.path.join(args.video_path, video)
        # pdb.set_trace()
        # name_vid_list = os.listdir(name)
        # for video in name_vid_list:
            # name_video = os.path.join(name, video)
        # pdb.set_trace()
        dst = os.path.join(args.out_dir, video[:-4])
        if not os.path.exists(dst):
            os.makedirs(dst)
        if len(os.listdir(dst)) == 0:
            extract_frames(name_video, dst)
            print("finish video id: " + video)






# 所有视频放一个文件夹里 
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--out_dir', dest='out_dir', type=str, default='data/LLP_dataset/frame')
#     parser.add_argument('--video_path', dest='video_path', type=str, default='data/LLP_dataset/video')
#     args = parser.parse_args()

#     vid_list = os.listdir(args.video_path)

#     for vid_id in vid_list:
#         name = os.path.join(args.video_path, vid_id)
#         dst = os.path.join(args.out_dir, vid_id[:-4])
#         print(dst)
#         if not os.path.exists(dst):
#             os.makedirs(dst)
#         extract_frames(name, dst)
#         print("finish video id: " + vid_id)