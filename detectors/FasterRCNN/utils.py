import os
import shutil


def video_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def xml_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def video_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def is_xml(filename):
    return any(filename.endswith(ext) for ext in '.xml')


def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.jpg','.png','JPG','.PNG'])



def remove_string(filenames,string):
    for i, filename in enumerate(filenames):
        if string in filename:
            del filenames[i]
    return filenames

def makedir(path):
    if os.path.exists(os.path.join(path)) == False:
        os.mkdir(path)
        print('make :',path)
    else :
        print('{} already exists.'.format(path))

def rmdir(path):
    if os.path.exists(os.path.join(path)) == False:
        pass
    else :
        shutil.rmtree(path)

def copy(file,path):
    shutil.copy(file,path)