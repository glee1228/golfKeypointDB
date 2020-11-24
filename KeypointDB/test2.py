import json
import os
from utils import makedir, image_basename, image_path, is_image
import copy
import math

images_root = '/home/mmlab/CCTV_Server/datasets/golfDB_18pts_200_test/train'
anchor_annotations_root = '/home/mmlab/CCTV_Server/datasets/golfDB_18pts_200/annotations'
result_annotations_root = '/home/mmlab/CCTV_Server/datasets/golfDB_18pts_200_test/annotations'

anchor_json_filename = 'golfDB_18pts_train_200.json'
result_json_filename = 'golfDB_18pts_train_200.json'
output_json_filename = 'golfDB_18pts_train_200_vis.json'

anchor_json_path = os.path.join(anchor_annotations_root,anchor_json_filename)
result_json_path = os.path.join(result_annotations_root,result_json_filename)
output_json_path = os.path.join(result_annotations_root,output_json_filename)

filenames = [image_basename(f)
             for f in os.listdir(images_root) if is_image(f)]

file_ids = []
for filename in filenames:
    file_ids.append(int(filename))

print(file_ids)
anchor_json_data = None
result_json_data = None

with open(anchor_json_path) as json_file:
    anchor_json_data = json.load(json_file)
    anchor_json_images = anchor_json_data['images']
    anchor_json_annotations = anchor_json_data['annotations']

with open(result_json_path) as json_file:
    result_json_data = json.load(json_file)
    result_json_images = result_json_data['images']
    result_json_annotations = result_json_data['annotations']


for i,anchor_json_annotation in enumerate(anchor_json_annotations):
    compare_idx = []
    anchor_id = anchor_json_annotation['image_id']
    anchor_keypoints = anchor_json_annotation['keypoints']
    anchor_left_hip_x,anchor_left_hip_y = anchor_keypoints[11*3],anchor_keypoints[11*3+1]
    anchor_right_hip_x, anchor_right_hip_y = anchor_keypoints[12 * 3], anchor_keypoints[12 * 3 + 1]
    anchor_center_x = anchor_left_hip_x-anchor_right_hip_x if anchor_left_hip_x>anchor_right_hip_x else anchor_right_hip_x-anchor_left_hip_x
    anchor_center_y =anchor_left_hip_y-anchor_right_hip_y if anchor_left_hip_y>anchor_right_hip_y else anchor_right_hip_y-anchor_left_hip_y
    is_club_head = anchor_keypoints[-1]

    for j,result_json_annotation in enumerate(result_json_annotations):
        result_id = result_json_annotation['image_id']
        if anchor_id == result_id :
            compare_idx.append(j)

    min_dist =1000000000
    min_idx = None

    for j in compare_idx:
        result_id = result_json_annotations[j]['image_id']
        result_keypoints = result_json_annotations[j]['keypoints']
        result_left_hip_x, result_left_hip_y = result_keypoints[11 * 3], result_keypoints[11 * 3 + 1]
        result_right_hip_x, result_right_hip_y = result_keypoints[12 * 3], result_keypoints[12 * 3 + 1]
        result_center_x = result_left_hip_x - result_right_hip_x if result_left_hip_x > result_right_hip_x else result_right_hip_x - result_left_hip_x
        result_center_y = result_left_hip_y - result_right_hip_y if result_left_hip_y > result_right_hip_y else result_right_hip_y - result_left_hip_y

        is_crowd = result_json_annotations[j]['iscrowd']

        if is_crowd ==0 :
            if is_club_head==2:
                result_json_annotations[j]['bbox']=anchor_json_annotations[i]['bbox']
                result_json_annotations[j]['keypoints'][-3:] = anchor_json_annotations[i]['keypoints'][-3:]
                result_json_annotations[j]['num_keypoints']+=1
            else:
                pass
        else:
                dist = math.sqrt(pow(anchor_center_x - result_center_x, 2) + pow(anchor_center_y - result_center_y, 2))
                if min_dist>dist:
                    min_dist=dist
                    min_idx = j

    if min_idx and is_club_head==2:
        result_json_annotations[min_idx]['bbox'] = anchor_json_annotations[i]['bbox']
        result_json_annotations[min_idx]['keypoints'][-3:]=anchor_json_annotations[i]['keypoints'][-3:]
        result_json_annotations[min_idx]['num_keypoints'] += 1
    elif min_idx and is_club_head==0:
        result_json_annotations[min_idx]['bbox'] = anchor_json_annotations[i]['bbox']


result_json_data['annotations']=result_json_annotations
with open(result_json_path, 'w') as outfile:
    json.dump(result_json_data, outfile)


for i,result_json_annotation in enumerate(result_json_annotations):
    keypoints = result_json_annotation['keypoints']
    keypoints_x, keypoints_y, keypoints_conf = [], [], []
    keypoints_list = []

    for k in range(0, len(keypoints), 3):
        keypoints_x.append(keypoints[k])
        keypoints_y.append(keypoints[k + 1])
        keypoints_conf.append(keypoints[k + 2])

    for pt_x, pt_y, conf in zip(keypoints_x, keypoints_y, keypoints_conf):
        if pt_x==0.0 and pt_y ==0.0:
            visibility = 0
        else:
            pt_x = pt_x
            pt_y = pt_y
            visibility = 2

        keypoints_list.append(pt_x)
        keypoints_list.append(pt_y)
        keypoints_list.append(visibility)

    result_json_annotations[i]['keypoints'] = keypoints_list

result_json_data['annotations']=result_json_annotations

with open(output_json_path, 'w') as outfile:
    json.dump(result_json_data, outfile)