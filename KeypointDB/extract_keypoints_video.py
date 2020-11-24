import os
import sys
import argparse
import ast
import cv2
import time
import torch
from vidgear.gears import CamGear
import numpy as np
from PIL import ImageDraw, Image
import random
sys.path.insert(1, os.getcwd())

from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations
import json
import cv2
import numpy as np
import torch
import os
from torchvision.transforms import transforms
from models.hrnet import HRNet
from models.poseresnet import PoseResNet
from PIL import Image
from time import gmtime, strftime
from utils import video_basename, json_basename, is_json,makedir

def create_heatmap(im_map,resized_x,resized_y,colormap=cv2.COLORMAP_HOT ):
    '''
    img is numpy array
    kernel_size must be odd ie. (5,5)
    '''

    # create blur image, kernel must be an odd number
    #im_cloud_blur = cv2.GaussianBlur(im_cloud,kernel_size,0)

    # If you need to invert the black/white data image
    # im_blur = np.invert(im_blur)
    # Convert back to BGR for cv2
    #im_cloud_blur = cv2.cvtColor(im_cloud_blur,cv2.COLOR_GRAY2BGR)
    # im_map *= 255
    #im_map = im_map.astype(np.uint8)
    # Apply colormap


    im_map[im_map<0]=0.0
    im_map = (im_map - np.min(im_map))/(np.max(im_map)-np.min(im_map))
    im_map *= 255
    im_map = im_map.astype('uint8')
    im_map = cv2.resize(im_map,(resized_x,resized_y),cv2.INTER_LINEAR)
    im_cloud_clr = cv2.applyColorMap(im_map, colormap)

    # blend images 50/50
    # return (a1*im_map + a2*im_cloud_clr).astype(np.uint8)
    return im_cloud_clr

class OnlySimpleHRNet:
    """
    SimpleHRNet class.

    The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
    weights, and predict the human pose on single images.
    Multi-person support with the YOLOv3 detector is also included (and enabled by default).
    """

    def __init__(self,
                 c,
                 nof_joints,
                 checkpoint_path,
                 model_name='HRNet',
                 resolution=(384, 288),
                 interpolation=cv2.INTER_CUBIC,
                 return_bounding_boxes=False,
                 max_batch_size=32,
                 device=torch.device("cpu")):
        """
        Initializes a new SimpleHRNet object.
        HRNet (and YOLOv3) are initialized on the torch.device("device") and
        its (their) pre-trained weights will be loaded from disk.

        Args:
            c (int): number of channels (when using HRNet model) or resnet size (when using PoseResNet model).
            nof_joints (int): number of joints.
            checkpoint_path (str): path to an official hrnet checkpoint or a checkpoint obtained with `train_coco.py`.
            model_name (str): model name (HRNet or PoseResNet).
                Valid names for HRNet are: `HRNet`, `hrnet`
                Valid names for PoseResNet are: `PoseResNet`, `poseresnet`, `ResNet`, `resnet`
                Default: "HRNet"
            resolution (tuple): hrnet input resolution - format: (height, width).
                Default: (384, 288)
            interpolation (int): opencv interpolation algorithm.
                Default: cv2.INTER_CUBIC
            multiperson (bool): if True, multiperson detection will be enabled.
                This requires the use of a people detector (like YOLOv3).
                Default: True
            return_bounding_boxes (bool): if True, bounding boxes will be returned along with poses by self.predict.
                Default: False
            max_batch_size (int): maximum batch size used in hrnet inference.
                Useless without multiperson=True.
                Default: 16
            yolo_model_def (str): path to yolo model definition file.
                Default: "./models/detectors/yolo/config/yolov3.cfg"
            yolo_class_path (str): path to yolo class definition file.
                Default: "./models/detectors/yolo/data/coco.names"
            yolo_weights_path (str): path to yolo pretrained weights file.
                Default: "./models/detectors/yolo/weights/yolov3.weights.cfg"
            device (:class:`torch.device`): the hrnet (and yolo) inference will be run on this device.
                Default: torch.device("cpu")
        """

        self.c = c
        self.nof_joints = nof_joints
        self.detector_root = '/home/mmlab/CCTV_Server/models/detectors'
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.interpolation = interpolation
        self.return_bounding_boxes = return_bounding_boxes
        self.max_batch_size = max_batch_size
        self.device = device
        self.previous_out_shape = None
        self.heatmap_club_head_cnt = 0
        self.heatmap_left_wrist_cnt = 0
        self.heatmap_club_head_dir = '/home/mmlab/CCTV_Server/golf/heatmap_club_head'
        self.heatmap_left_wrist_dir = '/home/mmlab/CCTV_Server/golf/heatmap_left_wrist'
        makedir(self.heatmap_club_head_dir)
        makedir(self.heatmap_left_wrist_dir)

        if model_name in ('HRNet', 'hrnet'):
            self.model = HRNet(c=c, nof_joints=nof_joints)
        elif model_name in ('PoseResNet', 'poseresnet', 'ResNet', 'resnet'):
            self.model = PoseResNet(resnet_size=c, nof_joints=nof_joints)
        else:
            raise ValueError('Wrong model name.')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]
            print(device_ids)

            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resolution[0], self.resolution[1])),  # (height, width)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def predict(self, image):
        """
        Predicts the human pose on a single image or a stack of n images.

        Args:
            image (:class:`np.ndarray`):
                the image(s) on which the human pose will be estimated.

                image is expected to be in the opencv format.
                image can be:
                    - a single image with shape=(height, width, BGR color channel)
                    - a stack of n images with shape=(n, height, width, BGR color channel)

        Returns:
            :class:`np.ndarray`:
                a numpy array containing human joints for each (detected) person.

                Format:
                    if image is a single image:
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).
                    if image is a stack of n images:
                        list of n np.ndarrays with
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).

                Each joint has 3 values: (y position, x position, joint confidence).

                If self.return_bounding_boxes, the class returns a list with (bounding boxes, human joints)
        """

        if len(image.shape) == 3:
            return self._predict_single(image)
        else:
            raise ValueError('Wrong image format.')

    def _predict_single(self, image):
        image = Image.fromarray(image)

        boxes = np.empty((1, 4), dtype=np.int32)
        images = torch.empty((1, 3, self.resolution[0], self.resolution[1]))  # (height, width)
        image = np.array(image)
        x1 = 0
        x2 = image.shape[1]
        y1 = 0
        y2 = image.shape[0]

        # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
        correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)

        if correction_factor > 1:
            # increase y side
            center = y1 + (y2 - y1) // 2
            length = int(round((y2 - y1) * correction_factor))
            y1 = max(0, center - length // 2)
            y2 = min(image.shape[0], center + length // 2)
        elif correction_factor < 1:
            # increase x side
            center = x1 + (x2 - x1) // 2
            length = int(round((x2 - x1) * 1 / correction_factor))
            x1 = max(0, center - length // 2)
            x2 = min(image.shape[1], center + length // 2)

        boxes[0] = [x1, y1, x2, y2]
        images[0] = self.transform(image[y1:y2,x1:x2,  ::-1])


        if images.shape[0] > 0:  # HRNet inference when there is more than one person
            images = images.to(self.device)

            with torch.no_grad():
                if len(images) <= self.max_batch_size:
                    out = self.model(images)

                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                        device=self.device
                    )
                    for i in range(0, len(images), self.max_batch_size):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])
            self.previous_out_shape = out.shape
            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            # For each human, for each joint: y, x, confidence
            for i, human in enumerate(out):
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))

                    # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                    # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                    # 2: confidences
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]
                    if j == 9:
                        heatmap = create_heatmap(joint,self.resolution[1],self.resolution[0])
                        cv2.imwrite(os.path.join(self.heatmap_left_wrist_dir,'%05d.png'%self.heatmap_left_wrist_cnt),heatmap)
                        self.heatmap_left_wrist_cnt+=1
                    if j == 17:
                        heatmap = create_heatmap(joint,self.resolution[1],self.resolution[0])
                        cv2.imwrite(os.path.join(self.heatmap_club_head_dir,'%05d.png'%self.heatmap_club_head_cnt),heatmap)
                        self.heatmap_club_head_cnt+=1


        else:
            pts = np.empty((0, 0, 3), dtype=np.float32)

        if self.return_bounding_boxes:
            return boxes, pts
        else:
            return pts


def live(camera_id, filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution,
       disable_tracking, max_batch_size, disable_vidgear, save_heatmap_video, video_format,
         video_framerate, device):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # print(device)
    if save_heatmap_video : print('save video.')
    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    if filename is not None:
        rotation_code = check_video_rotation(filename)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
    else:
        rotation_code = None
        if disable_vidgear:
            video = cv2.VideoCapture(camera_id)
            assert video.isOpened()
        else:
            video = CamGear(camera_id).start()

    model = OnlySimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        max_batch_size=max_batch_size,
        return_bounding_boxes=True,
        device=device
    )
    nof_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if not disable_tracking:
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0

    ############################# MAKE JSON FORMAT #####################################

    json_data = {}
    json_data['videos'] = []
    json_data['annotations'] = []
    json_data['categories'] = []

    frame_idx = 0


    while True:
        t = time.time()

        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
        else:
            frame = video.read()
            if frame is None:
                break
        _ = None
        if frame_idx==0:
            video_dict = {}
            video_dict['filename'] = str(filename)
            video_dict['height'] = frame.shape[0]
            video_dict['width'] = frame.shape[1]
            video_dict['date_captured'] = str(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            json_data['videos'].append(video_dict)

        pts = model.predict(frame)

        if not disable_tracking:
            boxes, pts = pts
        # import pdb;pdb.set_trace()
        object_idx = 0
        if len(boxes)==0:
            annotation_dict = {}
            annotation_dict['filename'] = str(filename)
            annotation_dict['num_keypoints'] = 0  # number of keypoints
            annotation_dict['area'] = 0.0  # w*h area
            annotation_dict['iscrowd'] = 0  # 0 : one person , 1 : more than one person
            annotation_dict['keypoints'] = [0.0 for i in range(0,54)]  # if 18 keypoints : number of points is 54
            annotation_dict['frame_id'] = int(frame_idx)
            annotation_dict['bbox'] = [0.0, 0.0, 0.0, 0.0]
            annotation_dict['category_id'] = 1
            annotation_dict['object_id'] = 0
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
                confidences = [conf for y,x,conf in pt]
                keypoints = list()
                num_keypoints = 0
                iscrowd = 0
                if len(pts)>1:
                    iscrowd =1
                for pt_x,pt_y,confidence in zip(keypoints_x,keypoints_y,confidences):
                    visibility = 0

                    if int(pt_x)==0 and int(pt_y)==0:
                        pt_x = 0
                        pt_y = 0
                        confidence =0
                    else :
                        pt_x = int(pt_x)
                        pt_y = int(pt_y)
                        num_keypoints+=1
                        confidence=float(confidence)
                    keypoints.append(pt_x)
                    keypoints.append(pt_y)
                    keypoints.append(confidence)


                annotation_dict = {}
                annotation_dict['filename'] = str(filename)
                annotation_dict['num_keypoints'] = num_keypoints  # number of keypoints
                annotation_dict['area'] = bbox_w*bbox_h  # w*h area
                annotation_dict['iscrowd'] = iscrowd  # 0 : one person , 1 : more than one person
                annotation_dict['keypoints'] = keypoints  # if 18 keypoints : number of points is 54
                annotation_dict['frame_id'] = int(frame_idx)
                annotation_dict['bbox'] = [bbox_x, bbox_y, bbox_w, bbox_h]
                annotation_dict['category_id'] = 1
                annotation_dict['object_id'] = int(object_idx)
                object_idx+=1
                json_data['annotations'].append(annotation_dict)
                # import pdb;pdb.set_trace()
        if save_heatmap_video:
            frame = cv2.imread('/home/mmlab/CCTV_Server/golf/heatmap_club_head/%05d.png'%frame_idx)

        frame_idx+=1
        fps = 1. / (time.time() - t)
        print('\rframe: % 4d / %d - framerate: %f fps ' % (frame_idx, nof_frames - 1, fps), end='')


        video_full_name = filename.split('/')[-1]

        output_root = '/home/mmlab/CCTV_Server/golf/output_heatmap'
        if frame_idx==1:
            makedir(output_root)
        output_path = os.path.join(output_root,video_full_name)


        if save_heatmap_video:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
                video_writer = cv2.VideoWriter(output_path, fourcc, video_framerate, (frame.shape[1], frame.shape[0]))

            video_writer.write(frame)

    if save_heatmap_video:
        video_writer.release()


    output_root = '/home/mmlab/CCTV_Server/golf/output_json'
    output_path = os.path.join(output_root,video_full_name)

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
                                                 [4, 6], [5, 7], [10, 18]]})

    file_basename = video_basename(video_full_name)
    json_output_filename = file_basename+'.json'
    assert json_output_filename.endswith('.json')
    with open(os.path.join(output_root, json_output_filename), "w") as json_file:
        json.dump(json_data, json_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default="/home/mmlab/CCTV_Server/golf/korea_videos/test4_data.mp4")
    parser.add_argument("--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=18)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="/home/mmlab/CCTV_Server/scripts/logs/20201016_0015/checkpoint_best_acc.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="golf")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')

    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--save_heatmap_video", help="save output frames into a video.", action="store_true",default=True)
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                                     "See http://www.fourcc.org/codecs.php", type=str, default='mp4v')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default='cuda:0')
    args = parser.parse_args()
    live(**args.__dict__)

