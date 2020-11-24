import json
import os
from utils import makedir, image_basename, image_path, is_image, json_basename

images_root = '/home/mmlab/CCTV_Server/datasets/golfDB_18pts_200_test/train'
annotations_root = '/home/mmlab/CCTV_Server/datasets/golfDB_18pts_200_test/annotations'

json_filename = 'golfDB_18pts_train_200_2.json'
json_base = json_basename(json_filename)
conf_thres = 0.4
output_json_filename = json_base+'_conf'+str(int(conf_thres*100))+'.json'
json_path = os.path.join(annotations_root,json_filename)
output_json_path = os.path.join(annotations_root,output_json_filename)


filenames = [image_basename(f)
             for f in os.listdir(images_root) if is_image(f)]

file_ids = []
for filename in filenames:
    file_ids.append(int(filename))

print(file_ids)
json_data = None

with open(json_path) as json_file:
    json_data = json.load(json_file)
    json_images = json_data['images']
    json_annotations = json_data['annotations']

for i,json_annotation in enumerate(json_annotations):
    visibility = 0
    keypoints = list(json_annotation['keypoints'])
    keypoints_x,keypoints_y,keypoints_conf = [],[],[]
    keypoints_list = []

    for k in range(0,len(keypoints),3):
        keypoints_x.append(keypoints[k])
        keypoints_y.append(keypoints[k+1])
        keypoints_conf.append(keypoints[k+2])

    num_keypoints=0
    for pt_x, pt_y, conf in zip(keypoints_x, keypoints_y, keypoints_conf):
        if conf>conf_thres:
            pt_x = pt_x
            pt_y = pt_y
            visibility = 2
            num_keypoints+=1
        else:
            pt_x = 0.0
            pt_y = 0.0
            visibility = 0

        keypoints_list.append(pt_x)
        keypoints_list.append(pt_y)
        keypoints_list.append(visibility)

    json_annotations[i]['keypoints']=keypoints_list
    json_annotations[i]['num_keypoints']=num_keypoints

json_data['annotations']=json_annotations

with open(output_json_path, 'w') as outfile:
    json.dump(json_data, outfile)