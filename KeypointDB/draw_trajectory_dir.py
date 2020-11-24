import os
from golf.draw_trajectory import draw_trajectory

def video_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def video_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def json_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def is_video(filename):
    return any(filename.endswith(ext) for ext in ['.mp4','.avi','.wmv','.MP4','.AVI','.WMV'])

def is_json(filename):
    return any(filename.endswith(ext) for ext in ['.json','.JSON'])

def makedir(path):
    if os.path.exists(os.path.join(path)) == False:
        os.mkdir(path)
        print('make :',path)
    else :
        print('{} already exists.'.format(path))


input_root = '/home/mmlab/CCTV_Server/golf/clip_videos'
json_root = '/home/mmlab/CCTV_Server/golf/output_json'
output_root = '/home/mmlab/CCTV_Server/golf/output_trajectory'

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
json_filenames = [f for f in os.listdir(json_root) if is_json(f)]
json_basenames = []
# print(json_filenames)
for json_filename in json_filenames:
    basename = json_basename(json_filename)
    json_basenames.append(basename)

for filename in filenames:
    basename = video_basename(filename)
    if basename in json_basenames:
        input_path = os.path.join(input_root,filename)
        json_path = os.path.join(json_root,basename+'.json')
        print()
        print(input_path)
        print(json_path)
        draw_trajectory(input_path, json_path,video_format,video_framerate)
