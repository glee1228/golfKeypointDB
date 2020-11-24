import os
import cv2
from tqdm import tqdm
import random
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


if __name__=='__main__':
    root_path = '/home/mmlab/CCTV_Server/data'
    input_root = 'golfDB_frames'
    output_root = 'golfkeypointDB'
    input_root = os.path.join(root_path,input_root)
    output_root = os.path.join(root_path,output_root)

    videonames = [f for f in os.listdir(input_root) if is_dir(os.path.join(input_root,f))]

    idx = 1000
    makedir(output_root)
    for videoname in tqdm(videonames):
        videopath = os.path.join(input_root,videoname)
        imgnames = os.listdir(videopath)
        imgnames = sorted(imgnames)
        if len(imgnames)>=30:
            imgnames = sorted(random.sample(imgnames,30))

        for imgname in imgnames:
            imgpath = os.path.join(videopath,imgname)
            img = cv2.imread(imgpath)
            cv2.imwrite(os.path.join(output_root,f'{idx:06d}.jpg'),img)
            idx+=1