import os
import pandas as pd
import cv2
import re
import time
from tqdm import tqdm

def make_clip(input_path,output_path,framerate,bbox,start_frame,end_frame):
    vidcap = cv2.VideoCapture(input_path)
    video_format = 'mp4v' # mp4v
    # print(input_path)
    start = start_frame
    end = end_frame
    x = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[0])
    y = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[1])
    w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[2])
    h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[3])
    nof_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    video_writer = None
    count = 0
    while vidcap.isOpened() :
        ret, frame = vidcap.read()
        t = time.time()
        if video_writer is None:
            # cropped = frame[y:y + h, x:x + w]
            fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
            video_writer = cv2.VideoWriter(output_path, fourcc, framerate, (frame.shape[1], frame.shape[0]))
        if start<=count and count<=end:
            # cropped = frame[y:y+h, x:x+w]
            video_writer.write(frame)
        elif count>=end:
            video_writer.release()
            vidcap.release()
        count += 1
        fps = 1. / (time.time() - t)
        # print('\rframe: % 4d / %d -  framerate: %0.1f fps ' % (count, nof_frames - 1, fps), end='')

    return count,nof_frames, fps


def video_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def video_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def is_video(filename):
    return any(filename.endswith(ext) for ext in ['.avi','.mp4','.mov','.mpeg','.mpg','.mkv','.flv','.wmf'])

def makedir(path):
    if os.path.exists(os.path.join(path)) == False:
        os.mkdir(path)
        print('make :',path)
    else :
        print('{} already exists.'.format(path))



if __name__=='__main__':
    root_path = '/home/mmlab/CCTV_Server/data'
    input_root = 'Youtube_golfDB'
    output_root = 'clip_videos'
    csv_path = 'golfDB.csv'
    input_root = os.path.join(root_path,input_root)
    output_root = os.path.join(root_path,output_root)
    csv_path = os.path.join(root_path,csv_path)

    framerate = 30
    filenames = [video_basename(f)
                 for f in os.listdir(input_root) if is_video(f)]

    df = pd.read_csv(csv_path)
    makedir(output_root)
    exist_video = False
    non_exists_video_filename = []
    pbar = tqdm(len(filenames), ncols=150)
    for idx,filename in enumerate(filenames):
        clip_video = None
        view = None
        count, nof_frames, fps = 0,0,0.0
        try:
            video_id=df.loc[df['title'] ==filename,'id'].values[0]
            view = df.loc[df['title']==filename,'view'].values[0]
            exist_video=True
        except :
            exist_video=False

        if exist_video and view =='face-on':
            input_path = None
            output_path = None
            input_path = video_path(input_root,filename,".mp4")
            output_path = video_path(output_root,filename+'_clip',".mp4")
            # input_path = input_path.replace(" ", "\ ").replace("&","\&").replace("(","\(").replace(")","\)")
            # output_path = output_path.replace(" ", "\ ").replace("&","\&").replace("(","\(").replace(")","\)")

            separators = "[", "]",'\'',' '
            regular_exp = '|'.join(map(re.escape, separators))

            bbox = [float(x) for x in re.split(regular_exp,df.loc[df['title']==filename,'bbox'].tolist()[0]) if x ]

            events = [float(x) for x in re.split(regular_exp,df.loc[df['title']==filename,'events'].tolist()[0]) if x ]
            start_frame = events[0]
            end_frame = events[-1]
            # print(f'{filename} : {start_frame} - {end_frame}')
            count, nof_frames, fps = make_clip(input_path,output_path,framerate,bbox,start_frame,end_frame)
            log = f'[index {idx}/{len(filenames)}] '
            log += f'frame: {count} / {nof_frames} -  framerate: {fps:.2f} fps  '
            pbar.set_description(log)
            pbar.update()

        else :
            non_exists_video_filename.append(filename)

    pbar.close()

    # print(non_exists_video_filename)


# w = 960
# h = 720
# x = int((1280-w)/2)
# y = int((720-h)/2)
#
# for filename in filenames:
#     input_path = None
#     output_path = None
#     input_path = video_path(input_root,filename,".mp4")
#     output_path = video_path(output_root,filename,".mp4")
#     input_path = input_path.replace(" ", "\ ")
#     output_path = output_path.replace(" ", "\ ")
#     # print(input_path)
#     # print(output_path)
#     command = 'yes | ffmpeg -i {} -vf "crop={}:{}:{}:{}" -strict experimental {}'.format(input_path,w,h,x,y,output_path)
#     print(command)
#     # import pdb;pdb.set_trace()
#     os.system(command)