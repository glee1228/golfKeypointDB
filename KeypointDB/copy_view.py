import os
import pandas as pd
import shutil

input_root = './frames_1280_720'
output_root = './images_1280_720'
csv_path = './golfDB.csv'
def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.jpg','.png','.bmp','.jpeg','.JPG','.PNG','.BMP'])

def is_dir(path):
    return os.path.isdir(path)

def makedir(path):
    if os.path.exists(os.path.join(path)) == False:
        os.mkdir(path)
        print('make :',path)
    else :
        print('{} already exists.'.format(path))


videonames = [f for f in os.listdir(input_root) if is_dir(os.path.join(input_root,f))]

margin_idx = 5
df = pd.read_csv(csv_path)
makedir(output_root)
view_dict = {'down-the-line':0,'face-on':1,'other':2,'None':3}

for videoname in videonames:
    view = ''
    try:
        view_value=df.loc[df['title'] ==videoname,'view'].values[0]
        view = view_dict[view_value]
    except :
        view = view_dict['None']

    if view == 1:
        filenames = [image_basename(f)
                     for f in os.listdir(os.path.join(input_root,videoname)) if is_image(f)]

        for filename in filenames:
            input_path = None
            output_path = None
            input_path = image_path(os.path.join(input_root,videoname), filename, '.jpg')
            output_path = image_path(output_root, filename,'.jpg')
            frame_idx = int(filename.split('_')[-1])
            video_id = int(filename.split('_')[-2])
            if frame_idx % margin_idx ==0:
                with open(input_path, 'rb') as fin:
                    with open(output_path, 'wb') as fout:
                        shutil.copyfileobj(fin, fout, 128 * 1024)



