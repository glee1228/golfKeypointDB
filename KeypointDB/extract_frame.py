import os
import pandas as pd
import cv2
import re
import time
from tqdm import tqdm
import random

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

def img_write(path,frame,img_id):
    cv2.imwrite(os.path.join(path,f'{img_id:06d}.jpg'),frame)
def make_frames(input_path,output_dir,framerate,bbox,events,slow_events,img_id):
    vidcap = cv2.VideoCapture(input_path)
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    start, address,toe_up, mid_backswing, top,mid_downswing,impact,mid_follow_through,finish, end = events
    if slow_events is not None:
        s_start, s_address, s_toe_up, s_mid_backswing, s_top, s_mid_downswing, s_impact, s_mid_follow_through, s_finish, s_end = slow_events
    # print(input_path)
    selected_frame = []
    slow_selected_frame = []
    start = 0
    for i, event in enumerate(events):
        if i==0:
            start=event
        else:
            inter_event_frame = range(int(start),int(event))
            if len(inter_event_frame)>=3:
                selected_frame.extend(random.sample(inter_event_frame,3))
            else:
                selected_frame.extend(random.sample(inter_event_frame, len(inter_event_frame)))
            start=event
    if slow_events is not None:
        for i, event in enumerate(slow_events):
            if i==0:
                start=event
            else:
                inter_event_frame = range(int(start),int(event))
                if len(inter_event_frame)>=3:
                    slow_selected_frame.extend(random.sample(inter_event_frame,3))
                else:
                    slow_selected_frame.extend(random.sample(inter_event_frame, len(inter_event_frame)))
                start=event

    x = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[0])
    y = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[1])
    w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[2])
    h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[3])
    nof_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0
    end_frame = max(end,s_end) if slow_events is not None else end

    while vidcap.isOpened() and count<=end_frame:
        ret, frame = vidcap.read()
        t = time.time()
        if frame is not None:
            if count in selected_frame:
                img_write(output_dir,frame,img_id)
                img_id += 1

            if slow_events is not None:
                if count in slow_selected_frame:
                    img_write(output_dir,frame,img_id)
                    img_id += 1

            count += 1
            fps = 1. / (time.time() - t)

        # print('\rframe: % 4d / %d -  framerate: %0.1f fps ' % (count, nof_frames - 1, fps), end='')
    return count, nof_frames, fps, img_id
    # except:
    #     print('error')
    #     return 0, 0, 0, img_id


def video_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def video_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def is_video(filename):
    return any(filename.endswith(ext) for ext in ['.avi','.mp4','.mov','.mpeg','.mpg','.mkv','.flv','.wmf'])

def makedir(path):
    if os.path.exists(os.path.join(path)) == False:
        os.mkdir(path)
        # print('make :',path)
    else :
        # print('{} already exists.'.format(path))
        pass

if __name__=='__main__':
    root_path = '/home/mmlab/CCTV_Server/data'
    input_root = 'Youtube_golfDB'
    output_root = 'clip_frames'
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
    img_id =1
    for idx,filename in enumerate(filenames):
        clip_video = None
        view,slow = None,None
        count, nof_frames, fps = 0,0,0.0
        try:
            video_id=df.loc[df['title'] ==filename,'id'].values[0]
            view = df.loc[df['title']==filename,'view'].values[0]
            exist_video=True
        except :
            exist_video=False

        if exist_video and view =='face-on':
            input_path = video_path(input_root,filename,".mp4")
            output_dir = os.path.join(output_root,filename)
            makedir(output_dir)
            events,slow_events=None,None
            separators = "[", "]",'\'',' '
            regular_exp = '|'.join(map(re.escape, separators))

            bbox = [float(x) for x in re.split(regular_exp,df.loc[df['title']==filename,'bbox'].tolist()[0]) if x ]
            try:
                events = [float(x) for x in re.split(regular_exp,df.loc[(df['title']==filename)&(df['slow']==0),'events'].tolist()[0]) if x ]
                slow_events = [float(x) for x in re.split(regular_exp,df.loc[(df['title']==filename)&(df['slow']==1),'events'].tolist()[0]) if x ]
            except:
                pass
            count, nof_frames, fps,img_id = make_frames(input_path,output_dir,framerate,bbox,events,slow_events,img_id)

            log = f'[index {idx}/{len(filenames)}] '
            log += f'filename : {filename[:50]} '
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