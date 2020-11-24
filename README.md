# GolfKeypointDB

#### 
1. `git clone https://github.com/glee1228/golfKeypointDB.git`

2. edit `docker-compose.yml`
    ```
    services:
      main:
        container_name: golfKeypointDB
        ...
        ports:
          - "{host ssh}:22"
          - "{host tensorboard}:6006"
        ipc: host
        stdin_open: true
    ```

3. `docker-compose up -d`

4. multi-person support: Faster R-CNN

    1. Install FasterRCNN required packages (Included in above requirements.txt)

    2. download COCO Weight files

       (my path : /mldisk/nfs_shared_/dh/golfKeypointDB/weights/faster_rcnn_obstacleV2.pth)

    3. Build `Non Maximum Suppression` and `ROI Align` modules (modified from [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)) 

       1. Install(`from folder /workspace/detectors/FasterRCNN`)

       ```
       $ python support/setup.py develop
       ```

       2. Uninstall

       ```
       $ python support/setup.py develop --uninstall
       ```

       3. Test

       ```
       $ python test/nms/test_nms.py
       ```

       - Result

         [![img](https://github.com/glee1228/ServerAnalysisModule/raw/master/models/detectors/FasterRCNN/images/test_nms.png?raw=true)](https://github.com/glee1228/ServerAnalysisModule/blob/master/models/detectors/FasterRCNN/images/test_nms.png?raw=true)

       - more details

       [potterhsu/easy-faster-rcnn.pytorch](https://github.com/potterhsu/easy-faster-rcnn.pytorch)

5. If you want to run the training script on COCO `main.py`, you have to build the `nms` module first.
    Please note that a linux machine with CUDA is currently required. Build it with either:
    
    - `cd misc; make` 
    
6.  Download the official pre-trained HR-Net weights from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch 

    (my path : /mldisk/nfs_shared_/dh/golfKeypointDB/weights/pose_hrnet_w48_384x288.pth)

    - COCO w48 384x288 (more accurate, but slower) 
      [pose_hrnet_w48_384x288.pth](https://drive.google.com/open?id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS)
    - COCO w32 256x192 (less accurate, but faster)
      [pose_hrnet_w32_256x192.pth](https://drive.google.com/open?id=1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38)
    - MPII w32 256x256 (MPII human joints)
      [pose_hrnet_w32_256x256.pth](https://drive.google.com/open?id=1_wn2ifmoQprBrFvUCDedjPON4Y6jsN-v)

7. Train  `main.py`

    ```bash
    #/workspace
    python main.py 
    ```
    Tensorboard
    ```bash
    # host에서 
    docker exec golfKeypointDB tensorboard --logdir={ckpt directory} --bind_all
    ```

    



 