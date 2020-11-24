import os
import sys
import argparse
import json
import ast
from utils import video_basename,makedir
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
import cv2
import time
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def sortFunction(value):
	return value["frame_id"]

def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

def draw_trajectory(filename, json_filename,video_format,video_framerate,save_frame,frame_dir):
    output_root = '/home/mmlab/CCTV_Server/golf/output_trajectory'
    makedir(output_root)
    with open(json_filename) as json_file:
        json_data = json.load(json_file)
        json_images = json_data['videos']
        json_annotations = json_data['annotations']

        # print('origin image len : ', len(json_images))
        # print('origin annotations len : ', len(json_annotations))

    sorted_annotations = sorted(json_annotations, key=sortFunction)
    # print(sorted_annotations)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    if filename is not None:
        rotation_code = check_video_rotation(filename)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()

    nof_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    prev_boxes = None
    prev_pts = None
    prev_person_ids = None
    next_person_id = 0

    frame_idx = 0
    points_list = []
    makedir(frame_dir)
    while frame_idx <nof_frames-1:
        t = time.time()
        annotation = sorted_annotations[frame_idx]
        # print(frame_idx, ' ' , annotation['frame_id'])
        # try:
        #     assert frame_idx is int(annotation['frame_id'])
        # except:
        #     print(frame_idx, ' ' ,int(annotation['frame_id']))
        #     import pdb;pdb.set_trace()

        if filename is not None :
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
        circle_size = max(1, min(frame.shape[:2]) // 160)


        pts = annotation['keypoints']

        pt_y,pt_x,conf = pts[51:54]
        # import pdb;pdb.set_trace()
        if conf > 0.75:
            points_list.append([pt_x,pt_y])
        prev_pt = None
        pprev_pt = None
        prev_i = None
        pprev_i = None
        save_p_i = 0
        cos = 0
        for p_i, pt in enumerate(points_list):
            # frame = cv2.circle(frame, (int(pt[1]), int(pt[0])), circle_size, (255,0,0), -1)
            save_p_i = p_i
            if p_i>=2 :
                # if euclidean_distance(prev_pt,pt)< int(frame.shape[0]//3):
                prev_mv = [p1 - p2 for p1,p2 in zip(prev_pt,pt)]
                pprev_mv = [p1 - p2 for p1,p2 in zip(pprev_pt,prev_pt)]
                if prev_mv[0] !=0 and prev_mv[1] !=0  and pprev_mv[0] !=0 and pprev_mv[1] !=0:
                    cos = cos_sim(pprev_mv,prev_mv)
                    if cos>=-0.5 and cos <=1:
                        frame = cv2.line(frame, (int(prev_pt[1]), int(prev_pt[0])), (int(pt[1]), int(pt[0])),
                                         (0, 0, 255), 3)
                    # else:
                    #     frame = cv2.line(frame, (int(prev_pt[1]), int(prev_pt[0])),
                    #                      (int(pt[1]), int(pt[0])),
                    #                      (0, 0, 255), 3)



                else :
                    pass
                if prev_i != p_i and prev_pt[0]!=pt[0] and prev_pt[1] != pt[1] :
                    if cos>=-0.5 and cos<=1:
                        pprev_pt = prev_pt
                        prev_pt = pt
                        pprev_i = prev_i
                        prev_i = p_i
            elif p_i == 0 :
                pprev_pt = pt
                pprev_i = p_i
            elif p_i ==1:
                prev_pt = pt
                prev_i = p_i


        if save_frame:
            if frame_idx>1546 and frame_idx<1600:
                frame_path = os.path.join(frame_dir,'%06d.jpg'%frame_idx)
                cv2.imwrite(frame_path, frame)

        fps = 1. / (time.time() - t)
        print('\rframe: % 4d / %d - framerate: %f fps  '% (frame_idx, nof_frames - 1, fps), end='')
        frame_idx+=1

        video_full_name = filename.split('/')[-1]
        output_path = os.path.join(output_root,video_full_name)

        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
            video_writer = cv2.VideoWriter(output_path, fourcc, video_framerate, (frame.shape[1], frame.shape[0]))

        video_writer.write(frame)


    video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", help="open the specified video ",
                        type=str, default="/home/mmlab/CCTV_Server/golf/korea_videos/test3.mp4")
    parser.add_argument("--json_filename", "-j", help="open the specified json ",
                        type=str, default="/home/mmlab/CCTV_Server/golf/output_json/test3.json")
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                               "See http://www.fourcc.org/codecs.php", type=str, default='mp4v')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
    parser.add_argument("--save_frame", help="saving frames", type=str, default=True)
    parser.add_argument("--frame_dir", "-s", help="open the specified json ",
                        type=str, default="/home/mmlab/CCTV_Server/golf/output_trajectory_frame")

    args = parser.parse_args()
    draw_trajectory(**args.__dict__)