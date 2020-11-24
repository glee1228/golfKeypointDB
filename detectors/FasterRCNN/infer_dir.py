import argparse
import os
import random
import torch

from PIL import ImageDraw
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from bbox import BBox
from model import Model
from roi.pooler import Pooler
from config.eval_config import EvalConfig as Config

from utils import image_basename, image_path, is_image, image_path


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
        parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))

        args = parser.parse_args()

        input_root = '/home/mmlab/CCTV_Server/models/detectors/FasterRCNN/frames'
        output_root = input_root+'_output'
        path_to_checkpoint = '/home/mmlab/CCTV_Server/models/detectors/FasterRCNN/checkpoints/obstacle/model-90000.pth'
        dataset_name = 'obstacle'
        backbone_name = 'resnet101'
        prob_thresh = 0.6
        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_sizes=args.anchor_sizes, pooler_mode=args.pooler_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n)

        print('Arguments:')
        for k, v in vars(args).items():
            print(f'\t{k} = {v}')
        print(Config.describe())

        os.makedirs(output_root,exist_ok=True)

        input_sub_dirnames = [directory for directory in os.listdir(input_root) if os.path.isdir(os.path.join(input_root,directory))]
        dataset_class = DatasetBase.from_name(dataset_name)
        backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
        model = Model(backbone, dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
                      anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                      rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
        model.load(path_to_checkpoint)

        for sub_dir in input_sub_dirnames:
            input_sub_dirpath = os.path.join(input_root,sub_dir)
            output_sub_dirpath = os.path.join(output_root,sub_dir)

            filenames = [image_basename(f)
                         for f in os.listdir(input_sub_dirpath) if is_image(f)]
            for filename in filenames:
                path_to_input_image = image_path(input_sub_dirpath,filename,'.jpg')
                # path_to_input_image = '/faster-RCNN/frames/1_360p/1_360p_0001.jpg'
                path_to_output_image = image_path(output_sub_dirpath,filename,'.jpg')
                # path_to_output_image = '/faster-RCNN/frames_output/1_360p/1_360p_0001.jpg'

                os.makedirs(os.path.join(os.path.curdir, os.path.dirname(path_to_output_image)), exist_ok=True)

                with torch.no_grad():
                    image = transforms.Image.open(path_to_input_image)
                    image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

                    detection_bboxes, detection_classes, detection_probs, _, _ = \
                        model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())
                    detection_bboxes /= scale

                    kept_indices = detection_probs > prob_thresh
                    detection_bboxes = detection_bboxes[kept_indices]
                    detection_classes = detection_classes[kept_indices]
                    detection_probs = detection_probs[kept_indices]

                    draw = ImageDraw.Draw(image)

                    for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(),
                                               detection_probs.tolist()):
                        color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
                        bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                        category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]

                        draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
                        draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

                    image.save(path_to_output_image)
                    print(f'Output image is saved to {path_to_output_image}')
    main()
