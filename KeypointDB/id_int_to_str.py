import json
import os
from utils import makedir, image_basename, image_path, is_image

images_root = '/home/mmlab/CCTV_Server/datasets/golfDB_18pts_200_test/train'
annotations_root = '/home/mmlab/CCTV_Server/datasets/golfDB_18pts_200_test/annotations'

json_filename = 'golfDB_18pts_train_200_2_conf90.json'

json_path = os.path.join(annotations_root,json_filename)



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


for i,json_image in enumerate(json_images):
    id = json_image['id']
    json_images[i]['id'] = int(id)

for i,json_annotation in enumerate(json_annotations):
        category_id = json_annotation['category_id']
        id = json_annotation['image_id']
        obj_id = json_annotation['id']
        json_annotations[i]['category_id']=int(category_id)
        json_annotations[i]['image_id'] = int(id)
        json_annotations[i]['id']=int(obj_id)


json_data['images']=json_images
json_data['annotations']=json_annotations

with open(json_path, 'w') as outfile:
    json.dump(json_data, outfile)