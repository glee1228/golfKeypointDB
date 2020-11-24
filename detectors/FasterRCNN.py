import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'detectors', 'FasterRCNN'))
import argparse
import os
import random
import torch
import numpy as np

from PIL import ImageDraw
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from bbox import BBox
from model import Model
from roi.pooler import Pooler
from config.eval_config import EvalConfig as Config


def _infer(path_to_input_image: str, path_to_output_image: str, path_to_checkpoint: str, dataset_name: str, backbone_name: str, prob_thresh: float):
    dataset_class = DatasetBase.from_name(dataset_name)
    backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
    model = FRCNN(backbone, dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
    model.load(path_to_checkpoint)

    with torch.no_grad():
        image = transforms.Image.open(path_to_input_image)
        image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

        detection_bboxes, detection_classes, detection_probs, _ = \
            model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())
        detection_bboxes /= scale

        kept_indices = detection_probs > prob_thresh
        detection_bboxes = detection_bboxes[kept_indices]
        detection_classes = detection_classes[kept_indices]
        detection_probs = detection_probs[kept_indices]

        draw = ImageDraw.Draw(image)

        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]

            draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
            draw.rectangle(((bbox.left+1, bbox.top+1), (bbox.right+1, bbox.bottom+1)), outline=color)
            draw.rectangle(((bbox.left+2, bbox.top+2), (bbox.right+2, bbox.bottom+2)), outline=color)
            draw.rectangle(((bbox.left+3, bbox.top+3), (bbox.right+3, bbox.bottom+3)), outline=color)
            draw.rectangle(((bbox.left+4, bbox.top+4), (bbox.right+4, bbox.bottom+4)), outline=color)
            draw.rectangle(((bbox.left+5, bbox.top+5), (bbox.right+5, bbox.bottom+5)), outline=color)

            draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

        image.save(path_to_output_image)
        print(f'Output image is saved to {path_to_output_image}')


class FRCNN:
    def __init__(self,path_to_checkpoint, dataset_name='obstacle', backbone_name='resnet101', prob_thresh=0.6):
        self.path_to_checkpoint = path_to_checkpoint
        self.dataset_name = dataset_name
        self.backbone_name = backbone_name
        self.prob_thresh = prob_thresh
        self.dataset_class = DatasetBase.from_name(dataset_name)
        self.backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
        # Set up model
        self.model =  Model(self.backbone, self.dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
                            anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                            rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()

        self.model.load(path_to_checkpoint)
        self.model.eval()  # Set in evaluation mode

    def predict_single(self, image):
        result = {}
        result['results'] = []
        with torch.no_grad():

            image_tensor, scale = self.dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

            detection_bboxes, detection_classes, detection_probs,intermediate_features, _ = \
                self.model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())

            detection_bboxes /= scale

            kept_indices = detection_probs > self.prob_thresh
            detection_bboxes = detection_bboxes[kept_indices]
            detection_classes = detection_classes[kept_indices]
            detection_probs = detection_probs[kept_indices]
            detection_results = []
            dummy = {}

            for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(),
                                       detection_probs.tolist()):
                detection_result = {}
                # color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
                # bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                category = self.dataset_class.LABEL_TO_CATEGORY_DICT[cls]
                #
                # draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
                # draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)
                label_dummy = []
                label = {}

                detection_result['position'] = {}
                label['description'] = category
                label['score'] = prob * 100
                label_dummy.append(label)
                detection_result['label'] = label_dummy
                detection_result['position']['h'] = int(bbox[3] - bbox[1])
                detection_result['position']['w'] = int(bbox[2] - bbox[0])
                detection_result['position']['y'] = int(bbox[1])
                detection_result['position']['x'] = int(bbox[0])

                detection_results.append(detection_result)
                # print(bbox,cls,prob)
            dummy['detection_result'] = detection_results
            result['results'].append(dummy)
            # print(json.dumps(result, indent=4, sort_keys=True))

            return result,intermediate_features

    # def predict_batch(self, image_tensor, scale):
    #     result = {}
    #     result['results'] = []
    #     with torch.no_grad():
    #         # import pdb;pdb.set_trace()
    #         # image_tensor, scale = self.dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
    #
    #         detection_bboxes, detection_classes, detection_probs, intermediate_features, detection_batch_indices = \
    #             self.model.eval().forward(image_tensor.cuda())
    #
    #
    #         scale_batch = scale[detection_batch_indices].unsqueeze(dim=-1).expand_as(detection_bboxes).to(
    #             device=detection_bboxes.device)
    #         detection_bboxes = detection_bboxes / scale_batch
    #
    #         kept_indices = (detection_probs > 0.05).nonzero().view(-1)
    #         detection_bboxes = detection_bboxes[kept_indices]
    #         detection_classes = detection_classes[kept_indices]
    #         detection_probs = detection_probs[kept_indices]
    #         detection_batch_indices = detection_batch_indices[kept_indices]
    #         batch_idx , detection_count = np.unique(detection_batch_indices,return_counts=True)
    #
    #         detection_results = []
    #         dummy = {}
    #         dummy['detection_result'] = None
    #         for i in range(0,batch_idx):
    #             result['results'].append*dummy)
    #         import pdb;pdb.set_trace()
    #
    #         for batch_i, detection_i in zip(batch_idx,detection_count):
    #
    #             for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(),
    #                                        detection_probs.tolist()):
    #                 detection_result = {}
    #                 # color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
    #                 # bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
    #                 category = self.dataset_class.LABEL_TO_CATEGORY_DICT[cls]
    #                 #
    #                 # draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
    #                 # draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)
    #                 label_dummy = []
    #                 label = {}
    #
    #                 detection_result['position'] = {}
    #                 label['description'] = category
    #                 label['score'] = prob * 100
    #                 label_dummy.append(label)
    #                 detection_result['label'] = label_dummy
    #                 detection_result['position']['h'] = int(bbox[3] - bbox[1])
    #                 detection_result['position']['w'] = int(bbox[2] - bbox[0])
    #                 detection_result['position']['y'] = int(bbox[1])
    #                 detection_result['position']['x'] = int(bbox[0])
    #
    #                 detection_results.append(detection_result)
    #                 # print(bbox,cls,prob)
    #
    #             dummy['detection_result'] = detection_results
    #             result['results'].append(dummy)
    #         # print(json.dumps(result, indent=4, sort_keys=True))
    #
    #         return result, intermediate_features

if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
        parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
        parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('-p', '--probability_threshold', type=float, default=0.6, help='threshold of detection probability')
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
        parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        parser.add_argument('input', type=str, help='path to input image')
        parser.add_argument('output', type=str, help='path to output result image')
        args = parser.parse_args()

        path_to_input_image = args.input
        path_to_output_image = args.output
        dataset_name = args.dataset
        backbone_name = args.backbone
        path_to_checkpoint = args.checkpoint
        prob_thresh = args.probability_threshold

        os.makedirs(os.path.join(os.path.curdir, os.path.dirname(path_to_output_image)), exist_ok=True)

        Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
                     anchor_ratios=args.anchor_ratios, anchor_sizes=args.anchor_sizes, pooler_mode=args.pooler_mode,
                     rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n)

        print('Arguments:')
        for k, v in vars(args).items():
            print(f'\t{k} = {v}')
        print(Config.describe())

        _infer(path_to_input_image, path_to_output_image, path_to_checkpoint, dataset_name, backbone_name, prob_thresh)

    main()
