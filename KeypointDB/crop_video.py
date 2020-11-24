import os

input_root = './Youtube_golfDB'
output_root = './cropped_video_960_720'
def video_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def video_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def is_video(filename):
    return any(filename.endswith(ext) for ext in ['.avi','.mp4','.mov','.mpeg','.mpg','.mkv','.flv','.wmf'])

filenames = [video_basename(f)
             for f in os.listdir(input_root) if is_video(f)]

w = 960
h = 720
x = int((1280-w)/2)
y = int((720-h)/2)

for filename in filenames:
    input_path = None
    output_path = None
    input_path = video_path(input_root,filename,".mp4")
    output_path = video_path(output_root,filename,".mp4")
    input_path = input_path.replace(" ", "\ ")
    output_path = output_path.replace(" ", "\ ")
    # print(input_path)
    # print(output_path)
    command = 'yes | ffmpeg -i {} -vf "crop={}:{}:{}:{}" -strict experimental {}'.format(input_path,w,h,x,y,output_path)
    print(command)
    # import pdb;pdb.set_trace()
    os.system(command)
