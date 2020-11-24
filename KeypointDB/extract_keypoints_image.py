import os
import sys
import ast
import cv2
import time
import torch
import json
sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import check_video_rotation
from datasets.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from time import gmtime, strftime
from utils import makedir, image_basename, image_path, is_image, json_basename

def main(input_root, output_root, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, image_resolution, single_person,
         max_batch_size, json_output_filename,conf_threshold, device):

    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    makedir(output_root)
    filenames = [image_basename(f) for f in os.listdir(input_root) if is_image(f)]

    image_resolution = ast.literal_eval(image_resolution)

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        max_batch_size=max_batch_size,
        return_bounding_boxes=True,
        device=device
    )

    json_data = {}
    json_data['info'] = {'description': 'golf keypoint DB', 'url': 'http://glee1228.tistory.com',
                         'version': '1.0', 'year': 2020, 'contributor': 'D.H. Lee , H.Y. Na,  S.W. Jang',
                         'date_created': '2020/11/18'}
    json_data['licenses'] = [{'url': 'https://creativecommons.org/licenses/by/3.0/legalcode', 'id': '1',
                              'name': 'CREATIVE COMMONS PUBLIC LICENSE'}]
    json_data['images'] = []
    json_data['annotations'] = []
    json_data['categories'] = []

    index = 0

    filenames = sorted(filenames)
    obj_id = 1
    image_ids = []
    for n_iter, filename in enumerate(filenames):
        t = time.time()
        image_id = int(filename)
        image_ids.append(image_id)

        image = cv2.imread(image_path(input_root,filename,'.jpg'))
        image_dict = {}
        image_dict['license'] = "1"
        image_dict['filename'] = str(filename) + '.jpg'
        image_dict['coco_url'] = 'http://163.239.25.33:8007/images/' + str(filename) + '.jpg'
        image_dict['height'] = image.shape[0]
        image_dict['width'] = image.shape[1]
        image_dict['date_captured'] = str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        image_dict['flickr_url'] = 'http://163.239.25.33:8007/images/' + str(filename) + '.jpg'
        image_dict['id'] = image_id
        json_data['images'].append(image_dict)

        boxes, pts, detections, intermediate_features = model.predict(image)


        if len(boxes)==0:
            annotation_dict = {}
            annotation_dict['segmentation'] = []
            annotation_dict['num_keypoints'] = 0  # number of keypoints
            annotation_dict['area'] = 0.0  # w*h area
            annotation_dict['iscrowd'] = 0  # 0 : one person , 1 : more than one person
            annotation_dict['keypoints'] = [0.0 for i in range(0,54)]  # if 18 keypoints : number of points is 54
            annotation_dict['image_id'] = int(image_id)
            annotation_dict['bbox'] = [0.0, 0.0, 0.0, 0.0]
            annotation_dict['category_id'] = 1
            annotation_dict['id'] = int(obj_id)
            obj_id+=1
            json_data['annotations'].append(annotation_dict)
        else:
            for idx, (box,pt) in enumerate(zip(boxes,pts)):
                # frame = Image.open('/home/mmlab/CCTV_Server/000000469067.jpg')
                # draw = ImageDraw.Draw(frame)
                # draw.rectangle(((0,184.8),(557.32,288.86+184.8)), outline='red')
                # # draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline='red')
                # frame.save(os.path.join(output_root,'test.jpg'))
                # import pdb;pdb.set_trace()
                bbox_x = round(float(box[0]),2)   # visipedia annotation tool x1,y1,x2,y2 bbox format
                bbox_y = round(float(box[1]),2)
                bbox_w = round(float(box[2]-box[0]),2)
                bbox_h = round(float(box[3]-box[1]),2)

                keypoints_x = [x for y,x,conf in pt]
                keypoints_y = [y for y,x,conf in pt]
                keypoints_conf = [conf for y,x,conf in pt]
                keypoints = list()
                num_keypoints = 0
                iscrowd = 0
                if len(pts)>1:
                    iscrowd =1
                for pt_x,pt_y,conf in zip(keypoints_x,keypoints_y,keypoints_conf):
                    pt_x = float(pt_x)
                    pt_y = float(pt_y)
                    num_keypoints+=1
                    conf=float(conf)

                    keypoints.append(pt_x)
                    keypoints.append(pt_y)
                    keypoints.append(conf)
                keypoints.append(0)
                keypoints.append(0)
                keypoints.append(0)

                annotation_dict = {}
                annotation_dict['segmentation'] = []
                annotation_dict['num_keypoints'] = num_keypoints  # number of keypoints
                annotation_dict['area'] = bbox_w*bbox_h  # w*h area
                annotation_dict['iscrowd'] = iscrowd  # 0 : one person , 1 : more than one person
                annotation_dict['keypoints'] = keypoints  # if 18 keypoints : number of points is 54
                annotation_dict['image_id'] = int(image_id)
                annotation_dict['bbox'] = [bbox_x, bbox_y, bbox_w, bbox_h]
                annotation_dict['category_id'] = 1
                annotation_dict['id'] = int(obj_id)
                obj_id+=1
                json_data['annotations'].append(annotation_dict)


        fps = 1. / (time.time() - t)
        print('\rframe: % 4d / %d - framerate: %f fps ' % (index, len(filenames) - 1, fps), end='')

        index += 1
    json_data['images']= sorted(json_data['images'], key=lambda k: k['id'])
    json_data['annotations'] = sorted(json_data['annotations'], key=lambda k: k['id'])
    print(image_ids)
    json_data['categories'].append({'supercategory': 'person',
                                    'id': '1',
                                    'name': 'person',
                                    'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                                  'left_shoulder',
                                                  'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
                                                  'right_wrist',
                                                  'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                                                  'right_ankle', 'club_head'],
                                    'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
                                                 [6, 7],
                                                 [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4],
                                                 [3, 5],
                                                 [4, 6], [5, 7],[10, 18]]})
    assert json_output_filename.endswith('.json')

    with open(os.path.join(output_root,json_output_filename), "w") as json_file:
        json.dump(json_data, json_file)


    file_ids = []
    for filename in filenames:
        file_ids.append(int(filename))

    print(file_ids)

    with open(os.path.join(output_root,json_output_filename)) as json_file:
        json_data = json.load(json_file)
        json_annotations = json_data['annotations']

    json_annotations = sorted(json_annotations, key=lambda k: k['id'])
    for i, json_annotation in enumerate(json_annotations):
        visibility = 0
        keypoints = list(json_annotation['keypoints'])
        keypoints_x, keypoints_y, keypoints_conf = [], [], []
        keypoints_list = []

        for k in range(0, len(keypoints), 3):
            keypoints_x.append(keypoints[k])
            keypoints_y.append(keypoints[k + 1])
            keypoints_conf.append(keypoints[k + 2])

        num_keypoints = 0
        for pt_x, pt_y, conf in zip(keypoints_x, keypoints_y, keypoints_conf):
            if conf > conf_threshold:
                pt_x = pt_x
                pt_y = pt_y
                visibility = 2
                num_keypoints += 1
            else:
                pt_x = 0.0
                pt_y = 0.0
                visibility = 0

            keypoints_list.append(pt_x)
            keypoints_list.append(pt_y)
            keypoints_list.append(visibility)

        json_annotations[i]['keypoints'] = keypoints_list
        json_annotations[i]['num_keypoints'] = num_keypoints

    json_data['annotations'] = json_annotations
    json_thres_output_filename = f'{json_output_filename[:-5]}_{int(conf_threshold*100)}.json'
    with open(os.path.join(output_root,json_thres_output_filename), 'w') as outfile:
        json.dump(json_data, outfile)

if __name__ == '__main__':
    input_root = '/home/mmlab/CCTV_Server/data/golfkeypointDB/train'
    output_root = '/home/mmlab/CCTV_Server/data/golfkeypointDB/annotations'

    hrnet_m = 'HRNet'
    hrnet_c = 48
    hrnet_j = 17
    conf_threshold = 0.5
    hrnet_weights = "/home/mmlab/CCTV_Server/weights/pose_hrnet_w48_384x288.pth"
    image_resolution = '(384, 288)'
    single_person = False
    max_batch_size = 1  # only 1.
    json_output_filename = 'golfKeypoint_train.json'
    device = 'cuda:0'
    num_workers = 4
    main(input_root, output_root, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, image_resolution, single_person,
         max_batch_size, json_output_filename,conf_threshold, device)

