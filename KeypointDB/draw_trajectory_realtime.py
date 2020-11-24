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
import numpy as np
import math

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def getAngle(L1, L2):
    ang = math.degrees(math.atan2(L1[1], L1[0]) - math.atan2(L2[1], L2[0]))
    return ang + 360 if ang < 0 else ang

def rotate_via_numpy(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    x, y = xy
    c, s = np.cos(radians), np.sin(radians)
    j = np.mat([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    return float(m.T[0]), float(m.T[1])

def sortFunction(value):
	return value["frame_id"]

def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

def sortFunction(value):
	return value["frame_id"]

def read_keypoints(path):
    with open(path) as json_file:
        json_data = json.load(json_file)

    json_annotations = json_data['annotations']
    sorted_json_annotations = sorted(json_annotations, key=sortFunction)
    pts = np.zeros(shape=(len(sorted_json_annotations),54))
    for i,json_annotation in enumerate(sorted_json_annotations):
        pts[i,:] = json_annotation['keypoints']
    pts = pts.reshape(len(sorted_json_annotations),18,3)
    return pts
def get_next_pt(frame_idx,club_pts,threshold):
    y,x,conf = club_pts[frame_idx]
    for i in range(frame_idx,len(club_pts)):
        next_y,next_x,next_conf = club_pts[i+1]
        if next_y-y!=0 and next_x-x!=0 and next_conf>threshold:
            return [next_y,next_x]

def draw_trajectory(filename, json_filename,save_frame,frame_dir,output_root):
    makedir(output_root)
    pts = read_keypoints(json_filename)
    club_pts = pts[:, 17, :]
    # left_wrist_pts = pts[:, 9, :]
    # left_hip_pts = pts[:, 11, :]
    # right_hip_pts = pts[:, 12, :]

    if filename is not None:
        rotation_code = check_video_rotation(filename)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()

    nof_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    prev_boxes = None
    prev_pts = None
    prev_person_ids = None
    next_person_id = 0
    dist = 0.0
    stop = False
    frame_idx = 0
    points_list = []
    makedir(frame_dir)
    while frame_idx <nof_frames-1:
        t = time.time()
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

        conf = club_pts[frame_idx,2]
        # import pdb;pdb.set_trace()
        if conf > 0.75:
            points_list.append(np.flip(club_pts[frame_idx,:2]))
        prev_pt = None
        pprev_pt = None
        prev_i = None
        pprev_i = None
        save_p_i = 0
        cos = 0
        for p_i, pt in enumerate(points_list):
            # frame = cv2.circle(frame, (int(pt[1]), int(pt[0])), circle_size, (255,0,0), -1)
            if p_i>=2 :
                # if euclidean_distance(prev_pt,pt)< int(frame.shape[0]//3):
                cur_mv = pt-prev_pt
                prev_mv = prev_pt-pprev_pt
                if cur_mv[0] !=0 and cur_mv[1] !=0  and prev_mv[0] !=0 and prev_mv[1] !=0:
                    dist = euclidean_distance(prev_pt,pt)
                    cos = cos_sim(prev_mv,cur_mv)
                    if cos>=-0.5 and cos <=1:
                        frame = cv2.line(frame, (int(prev_pt[1]), int(prev_pt[0])), (int(pt[1]), int(pt[0])),
                                     (0, 0, 255), 3)
                        save_p_i = p_i
                    # else:
                    #     frame = cv2.line(frame, (int(prev_pt[1]), int(prev_pt[0])), (int(pt[1]), int(pt[0])),
                    #                      (255, 0, 0), 3)





                else :
                    pass
                if prev_i != p_i and prev_pt[0]!=pt[0] and prev_pt[1] != pt[1] :
                    if cos>=-0.5 and cos<=1:
                        pprev_pt = prev_pt
                        prev_pt = pt
                        pprev_i = prev_i
                        prev_i = p_i
                    # else:
                    #     pprev_pt = prev_pt
                    #     prev_pt = pt
                    #     pprev_i = prev_i
                    #     prev_i = p_i
            elif p_i == 0 :
                pprev_pt = pt
                pprev_i = p_i
            elif p_i ==1:
                prev_pt = pt
                prev_i = p_i


        if save_frame:
            frame_path = os.path.join(frame_dir,'%06d.jpg'%save_p_i)
            cv2.imwrite(frame_path, frame)


        cv2.putText(frame, f'frame : {frame_idx}', (15, 25),cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 255), 2)
        cv2.putText(frame, f'dist : {dist:.2f}', (250, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)



        cv2.imshow('frame', frame)
        if stop:
            import pdb;
            pdb.set_trace()
        fps = 1. / (time.time() - t)
        print('\rframe: % 4d / %d - framerate: %f fps  '% (frame_idx, nof_frames - 1, fps), end='')
        frame_idx+=1

        end_flag = False
        if cv2.waitKey(10) & 0xFF == 32:
            while (1):
                if cv2.waitKey(10) & 0xFF == 32: break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    end_flag = True
                    break

        if cv2.waitKey(10) & 0xFF == ord('q') or end_flag:
            break




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", help="open the specified video ",
                        type=str, default="/home/mmlab/CCTV_Server/golf/korea_videos/test3.mp4")
    parser.add_argument("--json_filename", "-j", help="open the specified json ",
                        type=str, default="/home/mmlab/CCTV_Server/golf/output_json/test3.json")
    parser.add_argument("--save_frame", help="saving frames", type=str, default=True)
    parser.add_argument("--frame_dir", "-s", help="open the specified json ",
                        type=str, default="/home/mmlab/CCTV_Server/golf/output_trajectory_frame")
    parser.add_argument("--output_root","-o",type=str,default='/home/mmlab/CCTV_Server/golf/output_trajectory')

    args = parser.parse_args()
    filename = 'test2'

    args.filename = f"D:\\CCTV_Server\\golf\\korea_videos\\{filename}.mp4"
    args.json_filename = f'D:\\CCTV_Server\\golf\\output_json\\{filename}.json'
    args.frame_dir = f'D:\\CCTV_Server\\golf\\output_trajectory_frame'
    args.output_root = f'D:\\CCTV_Server\\golf\\output_trajectory'
    draw_trajectory(**args.__dict__)