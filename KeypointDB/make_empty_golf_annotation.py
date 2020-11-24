import json
import os
from utils import makedir, image_basename, image_path, is_image
import cv2
from time import gmtime, strftime

# json 파일은 같은 폴더에 있다고 가정!

images_root = '/home/mmlab/CCTV_Server/datasets/golfDB_18pts_200_test/train'

filenames = [image_basename(f)
             for f in os.listdir(images_root) if is_image(f)]

# for filename in filenames:
#     path_to_input_video = video_path(videos_root, filename, '.mp4')

json_data = {}
json_data['info']={'description': 'golfDB 2020 keypoints Dataset', 'url': 'http://glee1228.tistory.com', 'version': '1.0', 'year': 2020, 'contributor': 'GolfDB Consortium', 'date_created': '2020/10/15'}
json_data['licenses']=[{'url': 'https://creativecommons.org/licenses/by/3.0/legalcode', 'id': 1, 'name': 'CREATIVE COMMONS PUBLIC LICENSE'}]
json_data['images']=[]
json_data['annotations']=[]
json_data['categories']=[]


for filename in filenames:
    img = cv2.imread(image_path(images_root,filename,'.jpg'))
    image_dict = {}
    image_dict['license'] = 1
    image_dict['filename'] = filename+'.jpg'
    image_dict['coco_url'] = 'http://localhost:8007/images/'+filename+'.jpg'
    image_dict['height'] = img.shape[1]
    image_dict['width'] = img.shape[0]
    image_dict['date_captured'] = str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    image_dict['flickr_url'] = 'http://localhost:8007/images/'+filename+'.jpg'
    image_dict['id']= int(filename)
    json_data['images'].append(image_dict)

for i, filename in enumerate(filenames):
    img = cv2.imread(image_path(images_root,filename,'.jpg'))
    annotation_dict = {}
    annotation_dict['segmentation'] = []
    annotation_dict['num_keypoints'] = 0 # number of keypoints
    annotation_dict['area'] = 0.0 # w*h area
    annotation_dict['iscrowd'] = 0  # 0 : one person , 1 : more than one person
    annotation_dict['keypoints'] = [0 for i in range(0,54)] # if 18 keypoints : number of points is 54
    annotation_dict['image_id'] = int(filename)
    annotation_dict['bbox'] = [0.0,0.0,0.0,0.0]
    annotation_dict['category_id'] = 1
    annotation_dict['id'] = 0
    json_data['annotations'].append(annotation_dict)

json_data['categories'].append({'supercategory': 'person',
                                'id': 1,
                                'name': 'golf_person',
                                'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                                              'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                                              'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle','club_head'],
                                'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
                                             [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
                                             [4, 6], [5, 7], [10,18]]})

output_root = images_root.split('/')[:-1]
output_root = '/'.join(output_root)
output_root = os.path.join(output_root,'annotations')
json_filename = 'golfDB_18pts_train_200.json'

import pdb;pdb.set_trace()
with open(os.path.join(output_root,json_filename), 'w') as outfile:
    json.dump(json_data, outfile)

# with open('test/person_keypoints_val2017.json') as json_file:
#     json_data = json.load(json_file)
#
#     # Reset path for loading local images
#     for images in json_data['images']:
#         coco_url = images['coco_url']
#         images['coco_url'] = os.path.join('http://localhost:8007/images',coco_url.split('/')[-1])
#         images['flickr_url'] = os.path.join('http://localhost:8007/images',coco_url.split('/')[-1])
#
#     # Add Club_head catergory for labeling 18th keypoint
#     json_data['categories'][0]['keypoints'].append('club_head')
#
#     # Add visualization web page for 18th category tagging
#     for annotation in json_data['annotations']:
#         annotation['keypoints'].append(0)
#         annotation['keypoints'].append(0)
#         annotation['keypoints'].append(0)
#
#     import pdb;pdb.set_trace()
#     # with open('test/person_keypoints_val2017_ver2.json', 'w') as outfile:
#     #     json.dump(json_data, outfile)
