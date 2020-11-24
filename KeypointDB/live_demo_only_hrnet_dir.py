import os
from golf.live_demo_only_hrnet import live

def video_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def video_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def is_video(filename):
    return any(filename.endswith(ext) for ext in ['.mp4','.avi','.wmv','.MP4','.AVI','.WMV'])

def makedir(path):
    if os.path.exists(os.path.join(path)) == False:
        os.mkdir(path)
        print('make :',path)
    else :
        print('{} already exists.'.format(path))


input_root = '/home/mmlab/CCTV_Server/golf/clip_videos'
output_root = '/home/mmlab/CCTV_Server/golf/output'

makedir(output_root)
camera_id = 0
hrnet_m = 'HRNet'
hrnet_c = 48
hrnet_j = 18
hrnet_weights = "/home/mmlab/CCTV_Server/scripts/logs/20200928_0117/checkpoint_best_acc.pth"
hrnet_joints_set = 'golf'
image_resolution = '(384, 288)'
disable_tracking = False
max_batch_size = 16
disable_vidgear = True
save_video = True
video_format = 'mp4v'
video_framerate = 30
device = 'cuda:0'
filenames = [f for f in os.listdir(input_root) if is_video(f)]

for filename in filenames:
    input_path = os.path.join(input_root,filename)

    live(camera_id, input_path, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution,
       disable_tracking, max_batch_size, disable_vidgear, save_video, video_format,
         video_framerate, device)
