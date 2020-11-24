from scipy.io import loadmat
import pandas as pd
from pytube import YouTube
import json
import os
import time
import signal
import urllib

def video_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def video_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def is_video(filename):
    return any(filename.endswith(ext) for ext in ['.avi','.mp4','.mov','.mpeg','.mpg','.mkv','.flv','.wmf'])


class TimeoutError(Exception):
       def __init__(self, value = "Timed Out"):
           self.value = value
       def __str__(self):
           return repr(self.value)

def timeout(seconds_before_timeout):
       def decorate(f):
           def handler(signum, frame):
               raise TimeoutError()
           def new_f(*args, **kwargs):
               old = signal.signal(signal.SIGALRM, handler)
               signal.alarm(seconds_before_timeout)
               try:
                   result = f(*args, **kwargs)
               finally:
                   signal.signal(signal.SIGALRM, old)
               signal.alarm(0)
               return result
           # new_f.func_name = f.func_name
           new_f.__name__ = f.__name__
           return new_f
       return decorate

@timeout(60)
def download_youtube(youtube_id,path):
    youtube = YouTube('https://www.youtube.com/watch=?v={}'.format(youtube_id))
    # test = {}
    # test['videoDetails'] = youtube.player_response.get("videoDetails")
    # test['streamingData'] = youtube.player_response.get("streamingData")
    # test['microformat'] = youtube.player_response.get("microformat")

    # title = test['microformat']['playerMicroformatRenderer']['title']['simpleText']
    # import pdb;pdb.set_trace()
    title = youtube.title
    if title in 'BROOKE HENDERSON 120fps SLOW MOTION FAIRWAY IRON REAR GOLF SWING 1080 HD.mp4':
        print('BROOKE HENDERSON PASS')
        pass
    # print(json.dumps(test, indent=2))
    elif title in filenames:
        print('already exists .. ', title)
        pass
    else:
        print('downloading .. ', title)
        video = youtube.streams.get_highest_resolution()
        video.download(path)

x = loadmat('golfDB.mat')
l = list(x['golfDB'][0])
d = dict()
for idx, k in enumerate(l):
    d["{:3d}".format(idx)] = list(l[idx])
df = pd.DataFrame(d).T
df.columns = ["id","youtube_id","player", "sex", "club","view","slow","events","bbox","split"]

# data format cleansing
df['id'] = df['id'].apply(lambda x: x[0][0])
df['youtube_id'] = df['youtube_id'].apply(lambda x: x[0])
df['player'] = df['player'].apply(lambda x: x[0])
df['sex'] = df['sex'].apply(lambda x: x[0])
df['club'] = df['club'].apply(lambda x: x[0])
df['view'] = df['view'].apply(lambda x: x[0])
df['slow'] = df['slow'].apply(lambda x: x[0][0])
df['events'] = df['events'].apply(lambda x: x[0])
df['bbox'] = df['bbox'].apply(lambda x: x[0])
df['split'] = df['split'].apply(lambda x: x[0][0])

already_filename = []
download_path = '/golfdb/data/Youtube_golfDB'

filenames = [video_basename(f)
             for f in os.listdir(download_path) if is_video(f)]

youtube_ids = list(df['youtube_id'])
print('total youtube_id length in dataframe : {}'.format(len(youtube_ids)))
print(youtube_ids)

youtube_ids = sorted(list(set(youtube_ids))) # Deduplication
print('total origin video length : {}'.format(len(youtube_ids)))
print(youtube_ids)
error_url = []

for youtube_id in youtube_ids:
    try:
        download_youtube(youtube_id,download_path)
    except KeyError:
        error_url.append(youtube_id)
        print('Key Error url : {} '.format(youtube_id))
    except ValueError:
        error_url.append(youtube_id)
        print('Value Error url : {}'.format(youtube_id))
    except TimeoutError as e:
        error_url.append(youtube_id)
        print("stopped executing {} because {}".format(youtube_id,e))
    except urllib.request.HTTPError as e:
        error_url.append(youtube_id)
        print("stopped executing {} because {}".format(youtube_id,e))

print('error url list : ',error_url)