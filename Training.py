import numpy as np
import torch
from tqdm import tqdm

from Dataset import HumanPoseEstimationDataset
from misc.utils import flip_tensor, flip_back, get_final_preds

import os
from datetime import datetime

import tensorboardX as tb
import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from utils import JointsMSELoss, JointsOHKMMSELoss, AverageMeter
from misc.checkpoint import save_checkpoint, load_checkpoint
from misc.utils import flip_tensor, flip_back
from misc.visualization import save_images
from models import HRNet
import logging
import warnings
warnings.filterwarnings("ignore")

def init_logger(save_dir, comment=None):
    c_date, c_time = datetime.now().strftime("%Y%m%d/%H%M%S").split('/')
    if comment is not None:
        if os.path.exists(os.path.join(save_dir, c_date, comment)):
            comment += f'_{c_time}'
    else:
        comment = c_time
    log_dir = os.path.join(save_dir, c_date, comment)
    log_txt = os.path.join(log_dir, 'log.txt')

    os.makedirs(f'{log_dir}/ckpts')
    os.makedirs(f'{log_dir}/submissions')

    global logger
    logger = logging.getLogger(c_time)

    logger.setLevel(logging.INFO)
    logger = logging.getLogger(c_time)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    h_file = logging.FileHandler(filename=log_txt, mode='a')
    h_file.setFormatter(fmt)
    h_file.setLevel(logging.INFO)
    logger.addHandler(h_file)
    logger.info(f'Log directory ... {log_txt}')
    return log_dir

class Train(object):
    """
    Train  class.

    The class provides a basic tool for training HRNet.
    Most of the training options are customizable.

    The only method supposed to be directly called is `run()`.
    """

    def __init__(self,
                 exp_name,
                 ds_train,
                 ds_val,
                 epochs=210,
                 batch_size=16,
                 num_workers=4,
                 loss='JointsMSELoss',
                 lr=0.001,
                 lr_decay=True,
                 lr_decay_steps=(170, 200),
                 lr_decay_gamma=0.1,
                 optimizer='Adam',
                 weight_decay=0.,
                 momentum=0.9,
                 nesterov=False,
                 pretrained_weight_path=None,
                 checkpoint_path=None,
                 log_path='./logs',
                 use_tensorboard=True,
                 model_c=48,
                 model_nof_joints=18,
                 model_bn_momentum=0.1,
                 flip_test_images=True,
                 device=None
                 ):
        """
        Initializes a new Train object.

        The log folder is created, the HRNet model is initialized and optional pre-trained weights or saved checkpoints
        are loaded.
        The DataLoaders, the loss function, and the optimizer are defined.

        Args:
            exp_name (str):  experiment name.
            ds_train (HumanPoseEstimationDataset): train dataset.
            ds_val (HumanPoseEstimationDataset): validation dataset.
            epochs (int): number of epochs.
                Default: 210
            batch_size (int): batch size.
                Default: 16
            num_workers (int): number of workers for each DataLoader
                Default: 4
            loss (str): loss function. Valid values are 'JointsMSELoss' and 'JointsOHKMMSELoss'.
                Default: "JointsMSELoss"
            lr (float): learning rate.
                Default: 0.001
            lr_decay (bool): learning rate decay.
                Default: True
            lr_decay_steps (tuple): steps for the learning rate decay scheduler.
                Default: (170, 200)
            lr_decay_gamma (float): scale factor for each learning rate decay step.
                Default: 0.1
            optimizer (str): network optimizer. Valid values are 'Adam' and 'SGD'.
                Default: "Adam"
            weight_decay (float): weight decay.
                Default: 0.
            momentum (float): momentum factor.
                Default: 0.9
            nesterov (bool): Nesterov momentum.
                Default: False
            pretrained_weight_path (str): path to pre-trained weights (such as weights from pre-train on imagenet).
                Default: None
            checkpoint_path (str): path to a previous checkpoint.
                Default: None
            log_path (str): path where tensorboard data and checkpoints will be saved.
                Default: "./logs"
            use_tensorboard (bool): enables tensorboard use.
                Default: True
            model_c (int): hrnet parameters - number of channels.
                Default: 48
            model_nof_joints (int): hrnet parameters - number of joints.
                Default: 17
            model_bn_momentum (float): hrnet parameters - path to the pretrained weights.
                Default: 0.1
            flip_test_images (bool): flip images during validating.
                Default: True
            device (torch.device): device to be used (default: cuda, if available).
                Default: None
        """
        super(Train, self).__init__()

        self.exp_name = exp_name
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss = loss
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_gamma = lr_decay_gamma
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.pretrained_weight_path = pretrained_weight_path
        self.checkpoint_path = checkpoint_path
        self.log_path = os.path.join(log_path, self.exp_name)
        self.use_tensorboard = use_tensorboard
        self.model_c = model_c
        self.model_nof_joints = model_nof_joints
        self.model_bn_momentum = model_bn_momentum
        self.flip_test_images = flip_test_images
        self.epoch = 0


        os.makedirs(self.log_path, 0o755, exist_ok=True)  # exist_ok=False to avoid overwriting
        if self.use_tensorboard:
            self.summary_writer = tb.SummaryWriter(self.log_path)

        #
        # write all experiment parameters in parameters.txt and in tensorboard text field
        self.parameters = [x + ': ' + str(y) + '\n' for x, y in locals().items()]

        with open(os.path.join(self.log_path, 'parameters.txt'), 'w') as fd:
            fd.writelines(self.parameters)
        if self.use_tensorboard:
            self.summary_writer.add_text('parameters', '\n'.join(self.parameters))

        #
        # load model
        self.model = HRNet(c=self.model_c, nof_joints=self.model_nof_joints,
                           bn_momentum=self.model_bn_momentum).cuda()


        #
        # define loss and optimizers
        if self.loss == 'JointsMSELoss':
            self.loss_fn = JointsMSELoss()
        elif self.loss == 'JointsOHKMMSELoss':
            self.loss_fn = JointsOHKMMSELoss()
        else:
            raise NotImplementedError

        if optimizer == 'SGD':
            self.optim = SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                             momentum=self.momentum, nesterov=self.nesterov)
        elif optimizer == 'Adam':
            self.optim = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError


        # load pre-trained weights (such as those pre-trained on imagenet)
        if self.pretrained_weight_path is not None:
            if self.model_nof_joints == 18:
                pretrained_dict = torch.load(self.pretrained_weight_path)
                pretrained_dict_items = list(pretrained_dict.items())
                pretrained_model = {}
                j = 0
                for k, v in self.model.state_dict().items():
                    v = pretrained_dict_items[j][1]
                    k = pretrained_dict_items[j][0]

                    if k == 'final_layer.weight':
                        x = torch.rand(1,48,1,1).cuda()
                        v = torch.cat([v, x], dim=0)
                    if k == 'final_layer.bias':
                        x = torch.rand(1).cuda()
                        v = torch.cat([v,x],dim=0)
                    pretrained_model[k] = v
                    j +=1
                model_dict=self.model.state_dict()
                model_dict.update(pretrained_model)
                self.model.load_state_dict(model_dict,strict=True)
            else:
                self.model.load_state_dict(torch.load(self.pretrained_weight_path, strict=True))
            print('Pre-trained weights loaded.')

        self.model = nn.DataParallel(self.model.cuda())
        # self.model = nn.DataParallel(self.model.to(self.device))
        #
        # load previous checkpoint
        if self.checkpoint_path is not None:
            print('Loading checkpoint %s...' % self.checkpoint_path)
            if os.path.isdir(self.checkpoint_path):
                path = os.path.join(self.checkpoint_path, 'checkpoint_last.pth')
            else:
                path = self.checkpoint_path
            self.starting_epoch, self.model, self.optim, self.params = load_checkpoint(path, self.model, self.optim,
                                                                                       self.device)
        else:
            self.starting_epoch = 0

        if lr_decay:
            self.lr_scheduler = MultiStepLR(self.optim, list(self.lr_decay_steps), gamma=self.lr_decay_gamma,
                                            last_epoch=self.starting_epoch if self.starting_epoch else -1)

        #
        # load train and val datasets
        self.dl_train = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                                   num_workers=self.num_workers, drop_last=True)
        self.len_dl_train = len(self.dl_train)

        # dl_val = DataLoader(self.ds_val, batch_size=1, shuffle=False, num_workers=num_workers)
        self.dl_val = DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.len_dl_val = len(self.dl_val)

        #
        # initialize variables
        self.mean_loss_train = 0.
        self.mean_acc_train = 0.
        self.mean_loss_val = 0.
        self.mean_acc_val = 0.
        self.mean_mAP_val = 0.

        self.best_loss = None
        self.best_acc = None
        self.best_mAP = None

    def _train(self):
        self.model.train()

        for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_train, desc='Training')):
            image = image.cuda()
            target = target.cuda()
            target_weight = target_weight.cuda()

            self.optim.zero_grad()

            output = self.model(image)

            loss = self.loss_fn(output, target, target_weight)

            loss.backward()

            self.optim.step()

            # Evaluate accuracy
            # Get predictions on the input
            accs, avg_acc, cnt, joints_preds, joints_target = self.ds_train.evaluate_accuracy(output, target)

            self.mean_loss_train += loss.item()
            self.mean_acc_train += avg_acc.item()
            if self.use_tensorboard:
                self.summary_writer.add_scalar('train_loss', loss.item(),
                                               global_step=step + self.epoch * self.len_dl_train)
                self.summary_writer.add_scalar('train_acc', avg_acc.item(),
                                               global_step=step + self.epoch * self.len_dl_train)
                if step == 0:
                    save_images(image, target, joints_target, output, joints_preds, joints_data['joints_visibility'],
                                self.summary_writer, step=step + self.epoch * self.len_dl_train, prefix='train_')

        self.mean_loss_train /= len(self.dl_train)
        self.mean_acc_train /= len(self.dl_train)

        print('\nTrain: Loss %f - Accuracy %f' % (self.mean_loss_train, self.mean_acc_train))

    def _val(self):
        self.model.eval()

        with torch.no_grad():
            for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_val, desc='Validating')):
                image = image.cuda()
                target = target.cuda()
                target_weight = target_weight.cuda()

                output = self.model(image)

                if self.flip_test_images:
                    image_flipped = flip_tensor(image, dim=-1)
                    output_flipped = self.model(image_flipped)

                    output_flipped = flip_back(output_flipped, self.ds_val.flip_pairs)

                    output = (output + output_flipped) * 0.5

                loss = self.loss_fn(output, target, target_weight)

                # Evaluate accuracy
                # Get predictions on the input
                accs, avg_acc, cnt, joints_preds, joints_target = \
                    self.ds_train.evaluate_accuracy(output, target)

                self.mean_loss_train += loss.item()
                self.mean_acc_train += avg_acc.item()
                if self.use_tensorboard:
                    self.summary_writer.add_scalar('val_loss', loss.item(),
                                                   global_step=step + self.epoch * self.len_dl_train)
                    self.summary_writer.add_scalar('val_acc', avg_acc.item(),
                                                   global_step=step + self.epoch * self.len_dl_train)
                    if step == 0:
                        save_images(image, target, joints_target, output, joints_preds,
                                    joints_data['joints_visibility'], self.summary_writer,
                                    step=step + self.epoch * self.len_dl_train, prefix='val_')

        self.mean_loss_val /= len(self.dl_val)
        self.mean_acc_val /= len(self.dl_val)

        print('\nValidation: Loss %f - Accuracy %f' % (self.mean_loss_val, self.mean_acc_val))

    def _checkpoint(self):

        save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_last.pth'), epoch=self.epoch + 1, model=self.model,
                        optimizer=self.optim, params=self.parameters)

        if self.best_loss is None or self.best_loss > self.mean_loss_val:
            self.best_loss = self.mean_loss_val
            # print('best_loss %f at epoch %d' % (self.best_loss, self.epoch + 1))
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_loss.pth'), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optim, params=self.parameters)
        if self.best_acc is None or self.best_acc < self.mean_acc_val:
            self.best_acc = self.mean_acc_val
            # print('best_acc %f at epoch %d' % (self.best_acc, self.epoch + 1))
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_acc.pth'), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optim, params=self.parameters)
        if self.best_mAP is None or self.best_mAP < self.mean_mAP_val:
            self.best_mAP = self.mean_mAP_val
            # print('best_mAP %f at epoch %d' % (self.best_mAP, self.epoch + 1))
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_mAP.pth'), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optim, params=self.parameters)

    def run(self):
        """
        Runs the training.
        """

        print('\nTraining started @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # start training
        for self.epoch in range(self.starting_epoch, self.epochs):
            # print('\nEpoch %d of %d @ %s' % (self.epoch + 1, self.epochs, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            self.mean_loss_train = 0.
            self.mean_loss_val = 0.
            self.mean_acc_train = 0.
            self.mean_acc_val = 0.
            self.mean_mAP_val = 0.

            #
            # Train

            self._train()

            #
            # Val

            self._val()

            #
            # LR Update

            if self.lr_decay:
                self.lr_scheduler.step()

            #
            # Checkpoint

            self._checkpoint()

        print('\nTraining ended @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

class GOLFTrain(Train):
    """
    COCOTrain class.

    Extension of the Train class for the COCO dataset.
    """

    def __init__(self,
                 exp_name,
                 ds_train,
                 ds_val,
                 epochs=210,
                 batch_size=16,
                 num_workers=4,
                 loss='JointsMSELoss',
                 lr=0.001,
                 lr_decay=True,
                 lr_decay_steps=(170, 200),
                 lr_decay_gamma=0.1,
                 optimizer='Adam',
                 weight_decay=0.,
                 momentum=0.9,
                 nesterov=False,
                 pretrained_weight_path=None,
                 checkpoint_path=None,
                 log_path='./logs',
                 use_tensorboard=True,
                 model_c=48,
                 model_nof_joints=18,
                 model_bn_momentum=0.1,
                 flip_test_images=True,
                 device=None
                 ):
        """
        Initializes a new COCOTrain object which extends the parent Train class.
        The initialization function calls the init function of the Train class.

        Args:
            exp_name (str):  experiment name.
            ds_train (HumanPoseEstimationDataset): train dataset.
            ds_val (HumanPoseEstimationDataset): validation dataset.
            epochs (int): number of epochs.
                Default: 210
            batch_size (int): batch size.
                Default: 16
            num_workers (int): number of workers for each DataLoader
                Default: 4
            loss (str): loss function. Valid values are 'JointsMSELoss' and 'JointsOHKMMSELoss'.
                Default: "JointsMSELoss"
            lr (float): learning rate.
                Default: 0.001
            lr_decay (bool): learning rate decay.
                Default: True
            lr_decay_steps (tuple): steps for the learning rate decay scheduler.
                Default: (170, 200)
            lr_decay_gamma (float): scale factor for each learning rate decay step.
                Default: 0.1
            optimizer (str): network optimizer. Valid values are 'Adam' and 'SGD'.
                Default: "Adam"
            weight_decay (float): weight decay.
                Default: 0.
            momentum (float): momentum factor.
                Default: 0.9
            nesterov (bool): Nesterov momentum.
                Default: False
            pretrained_weight_path (str): path to pre-trained weights (such as weights from pre-train on imagenet).
                Default: None
            checkpoint_path (str): path to a previous checkpoint.
                Default: None
            log_path (str): path where tensorboard data and checkpoints will be saved.
                Default: "./logs"
            use_tensorboard (bool): enables tensorboard use.
                Default: True
            model_c (int): hrnet parameters - number of channels.
                Default: 48
            model_nof_joints (int): hrnet parameters - number of joints.
                Default: 17
            model_bn_momentum (float): hrnet parameters - path to the pretrained weights.
                Default: 0.1
            flip_test_images (bool): flip images during validating.
                Default: True
            device (torch.device): device to be used (default: cuda, if available).
                Default: None
        """
        super(GOLFTrain, self).__init__(
            exp_name=exp_name,
            ds_train=ds_train,
            ds_val=ds_val,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            loss=loss,
            lr=lr,
            lr_decay=lr_decay,
            lr_decay_steps=lr_decay_steps,
            lr_decay_gamma=lr_decay_gamma,
            optimizer=optimizer,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            pretrained_weight_path=pretrained_weight_path,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            use_tensorboard=use_tensorboard,
            model_c=model_c,
            model_nof_joints=model_nof_joints,
            model_bn_momentum=model_bn_momentum,
            flip_test_images=flip_test_images,
            device=device
        )

    def _train(self):

        num_samples = self.len_dl_train * self.batch_size
        all_preds = np.zeros((num_samples, self.model_nof_joints, 3), dtype=np.float32)
        all_boxes = np.zeros((num_samples, 6), dtype=np.float32)
        image_paths = []
        idx = 0

        self.model.train()
        losses = AverageMeter()
        avg_accs = AverageMeter()
        pbar = tqdm(self.dl_train, ncols=170)
        for step, (image, target, target_weight, joints_data) in enumerate(self.dl_train):
            image = image.cuda()
            target = target.cuda()
            target_weight = target_weight.cuda()

            self.optim.zero_grad()

            output = self.model(image)

            loss = self.loss_fn(output, target, target_weight)

            loss.backward()

            self.optim.step()

            # Evaluate accuracy
            # Get predictions on the resized images (given as input)
            accs, avg_acc, cnt, joints_preds, joints_target = \
                self.ds_train.evaluate_accuracy(output, target)

            losses.update(loss)
            avg_accs.update(avg_acc)
            # Original
            num_images = image.shape[0]

            # measure elapsed time
            c = joints_data['center'].numpy()
            s = joints_data['scale'].numpy()
            score = joints_data['score'].numpy()
            pixel_std = 200  # ToDo Parametrize this

            # Get predictions on the original imagee
            preds, maxvals = get_final_preds(True, output.detach(), c, s,
                                             pixel_std)  # ToDo check what post_processing exactly does

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2].detach().cpu().numpy()
            all_preds[idx:idx + num_images, :, 2:3] = maxvals.detach().cpu().numpy()
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * pixel_std, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_paths.extend(joints_data['imgPath'])

            idx += num_images

            log = f'[Epoch {self.epoch}] '
            log += f'Train loss : {loss.item():.4f}({losses.avg:.4f}) '
            log += f'Train acc : {avg_acc.item():.4f}({avg_accs.avg:.4f}) '

            pbar.set_description(log)
            pbar.update()

            self.mean_loss_train += loss.item()
            if self.use_tensorboard:
                self.summary_writer.add_scalar('Train/Loss', loss.item(),
                                               global_step=step + self.epoch * self.len_dl_train)
                self.summary_writer.add_scalar('Train/Accuracy', avg_acc.item(),
                                               global_step=step + self.epoch * self.len_dl_train)
                if step == 0:
                    save_images(image, target, joints_target, output, joints_preds, joints_data['joints_visibility'],
                                self.summary_writer, step=step + self.epoch * self.len_dl_train, prefix='train_')

        self.mean_loss_train /= len(self.dl_train)

        # COCO evaluation
        # print('\nTrain AP/AR')
        self.train_accs, self.mean_mAP_train = self.ds_train.evaluate_overall_accuracy(
            all_preds, all_boxes, image_paths, output_dir=self.log_path)


        mean_mAP = self.train_accs['AP'] #  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]
        AP_5 = self.train_accs['Ap .5'] # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ]
        AP_75 = self.train_accs['AP .75'] # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ]
        mean_mAR = self.train_accs['AR']  # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.378
        AR_5 = self.train_accs['AR .5'] # Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ]
        AR_75 = self.train_accs['AR .75'] # Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ]

        _lr = self.optim.param_groups[0]['lr']
        log = f'[EPOCH {self.epoch}] Train Loss : {losses.avg:.4f}, '
        log += f'Train acc : {avg_accs.avg:.4f}, '
        log += f'AP : {mean_mAP:.4f}, '
        log += f'AP.5 : {AP_5:.4f}, '
        log += f'AP.75 : {AP_75:.4f}, '
        log += f'AR : {mean_mAR:.4f}, '
        log += f'LR : {_lr:.2e}'
        pbar.set_description(log)
        pbar.close()

        if self.use_tensorboard:
            self.summary_writer.add_scalar('Train/mean_mAP', mean_mAP,
                                           global_step=step + self.epoch * self.len_dl_train)
            self.summary_writer.add_scalar('Train/AP.5', AP_5,
                                           global_step=step + self.epoch * self.len_dl_train)
            self.summary_writer.add_scalar('Train/AP.75', AP_75,
                                           global_step=step + self.epoch * self.len_dl_train)
            self.summary_writer.add_scalar('Train/mean_mAR', mean_mAR,
                                           global_step=step + self.epoch * self.len_dl_train)
            self.summary_writer.add_scalar('Train/AR.5', AR_5,
                                           global_step=step + self.epoch * self.len_dl_train)
            self.summary_writer.add_scalar('Train/AR.75', AR_75,
                                           global_step=step + self.epoch * self.len_dl_train)

    @torch.no_grad()
    def _val(self):
        num_samples = len(self.ds_val)

        all_preds = np.zeros((num_samples, self.model_nof_joints, 3), dtype=np.float32)
        all_boxes = np.zeros((num_samples, 6), dtype=np.float32)
        image_paths = []
        idx = 0
        self.model.eval()
        losses = AverageMeter()
        avg_accs = AverageMeter()
        pbar = tqdm(self.dl_val, ncols=170)

        for step, (image, target, target_weight, joints_data) in enumerate(self.dl_val):
            image = image.cuda()
            target = target.cuda()
            target_weight = target_weight.cuda()

            output = self.model(image)

            if self.flip_test_images:
                image_flipped = flip_tensor(image, dim=-1)
                output_flipped = self.model(image_flipped)

                output_flipped = flip_back(output_flipped, self.ds_val.flip_pairs)

                output = (output + output_flipped) * 0.5

            loss = self.loss_fn(output, target, target_weight)

            # Evaluate accuracy
            # Get predictions on the resized images (given as input)
            accs, avg_acc, cnt, joints_preds, joints_target = \
                self.ds_train.evaluate_accuracy(output, target)

            losses.update(loss)
            avg_accs.update(avg_acc)

            # Original
            num_images = image.shape[0]

            log = f'[Epoch {self.epoch}] '
            log += f'Valid loss : {loss.item():.4f}({losses.avg:.4f}) '
            log += f'Valid acc : {avg_acc.item():.4f}({avg_accs.avg:.4f}) '
            pbar.set_description(log)
            pbar.update()

            # measure elapsed time
            c = joints_data['center'].numpy()
            s = joints_data['scale'].numpy()
            score = joints_data['score'].numpy()
            pixel_std = 200  # ToDo Parametrize this

            preds, maxvals = get_final_preds(True, output, c, s,
                                             pixel_std)  # ToDo check what post_processing exactly does

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2].detach().cpu().numpy()
            all_preds[idx:idx + num_images, :, 2:3] = maxvals.detach().cpu().numpy()
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * pixel_std, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_paths.extend(joints_data['imgPath'])

            idx += num_images

            self.mean_loss_val += loss.item()
            self.mean_acc_val += avg_acc.item()
            if self.use_tensorboard:
                self.summary_writer.add_scalar('Valid/Loss', loss.item(),
                                               global_step=step + self.epoch * self.len_dl_val)
                self.summary_writer.add_scalar('Valid/Accuracy', avg_acc.item(),
                                               global_step=step + self.epoch * self.len_dl_val)
                if step == 0:
                    save_images(image, target, joints_target, output, joints_preds,
                                joints_data['joints_visibility'], self.summary_writer,
                                step=step + self.epoch * self.len_dl_train, prefix='test_')

        self.mean_loss_val /= len(self.dl_val)
        self.mean_acc_val /= len(self.dl_val)

        # COCO evaluation
        # print('\nVal AP/AR')
        self.val_accs, self.mean_mAP_val = self.ds_val.evaluate_overall_accuracy(
            all_preds, all_boxes, image_paths, output_dir=self.log_path)

        mean_mAP = self.val_accs['AP']  # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]
        AP_5 = self.val_accs['Ap .5']  # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ]
        AP_75 = self.val_accs['AP .75']  # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ]
        mean_mAR = self.val_accs[
            'AR']  # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.378
        AR_5 = self.val_accs['AR .5']  # Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ]
        AR_75 = self.val_accs['AR .75']  # Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ]

        log = f'[EPOCH {self.epoch}] Valid Loss : {losses.avg:.4f}, '
        log += f'Valid acc : {avg_accs.avg:.4f}, '
        log += f'AP : {mean_mAP:.4f}, '
        log += f'AP.5 : {AP_5:.4f}, '
        log += f'AP.75 : {AP_75:.4f}, '
        log += f'AR : {mean_mAR:.4f}, '
        pbar.set_description(log)
        pbar.close()

        if self.use_tensorboard:
            self.summary_writer.add_scalar('Valid/mean_mAP', mean_mAP,
                                           global_step=step + self.epoch * self.len_dl_val)
            self.summary_writer.add_scalar('Valid/AP.5', AP_5,
                                           global_step=step + self.epoch * self.len_dl_val)
            self.summary_writer.add_scalar('Valid/AP.75', AP_75,
                                           global_step=step + self.epoch * self.len_dl_val)
            self.summary_writer.add_scalar('Valid/mean_mAR', mean_mAR,
                                           global_step=step + self.epoch * self.len_dl_val)
            self.summary_writer.add_scalar('Valid/AR.5', AR_5,
                                           global_step=step + self.epoch * self.len_dl_val)
            self.summary_writer.add_scalar('Valid/AR.75', AR_75,
                                           global_step=step + self.epoch * self.len_dl_val)