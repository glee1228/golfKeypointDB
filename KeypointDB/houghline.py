import numpy as np
import cv2
import json
from datetime import datetime
import pandas as pd

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


json_filename = 'D:\\CCTV_Server\\golf\\output_json\\out.json'
cap = cv2.VideoCapture('D:\\CCTV_Server\\golf\\korea_videos\\out.mp4')
pts = read_keypoints(json_filename)
club_pts = pts[:,17,:]


# Take first frame and find corners in it
ret, frame = cap.read()

club_p0 = club_pts[0,:]
# Create a mask image for drawing purposes
mask = np.zeros_like(frame)

frame_idx = 0
is_club = False
# Assigning our static_back to None
static_back = None
prev_frame = None
pprev_frame = None
# List when any moving object appear
motion_list = [None, None]

# Time of movement
time = []
# Initializing DataFrame, one column is start
# time and other column is end time
df = pd.DataFrame(columns = ["Start", "End"])


while(1):
    ret,frame = cap.read()
    if not ret:
        break

    if club_pts[frame_idx, 2] > 0.75:
        is_club = True
    else:
        is_club = False

    if is_club and frame_idx==1:
        club_p1 = club_pts[frame_idx,:2]

    if is_club and frame_idx>1:
        club_p2 = club_pts[frame_idx,:2]
        club_p1 = club_p2
        club_p0 = club_p1
        frame = cv2.circle(frame, (int(club_p2[0]), int(club_p2[1])), 5, (255,0,0), -1)

    # Initializing motion = 0(no motion)
    motion = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Converting gray scale image to GaussianBlur
    # so that change can be find easily
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # In first iteration we assign the value
    # of static_back to our first frame
    if static_back is None:
        static_back = gray_blur
        continue


    # Difference between static background
    # and current frame(which is GaussianBlur)

    diff_frame = cv2.absdiff(static_back, gray_blur)

    # # If change in between static background and
    # # current frame is greater than 30 it will show white color(255)
    diff_frame = cv2.threshold(diff_frame, 20, 255, cv2.THRESH_BINARY)[1]
    diff_frame = cv2.dilate(diff_frame, None, iterations=2)
    #
    # # Finding contour of moving object
    _, cnts, _ = cv2.findContours(diff_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if frame_idx==20:
    #     import pdb;pdb.set_trace()
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        # making green rectangle arround the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Appending status of motion
    motion_list.append(motion)

    motion_list = motion_list[-2:]

    # Appending Start time of motion
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.now())

        # Appending End time of motion
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.now())

    # b = gray*a.copy().astype(bool)
    # canny =  cv2.Canny(diff_frame,100,200)
    canny = cv2.Canny(diff_frame, 5000, 1500, apertureSize=5, L2gradient=True)
    # Create default parametrization LSD
    good_lines = cv2.HoughLines(canny,1,np.pi/180,100)
    bad_lines =  cv2.HoughLines(canny,1,np.pi/180,50)
    if good_lines is not None:
        good_thetas = []
        for line in good_lines:
            for rho, theta in line:
                good_thetas.append(theta)

        val, count = np.unique(np.array(good_thetas),return_counts=True)
        good_parallel = val[count>=2]
        lines = []
        for line in good_lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                if theta in good_parallel:
                    frame = cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
                    frame = cv2.circle(frame, (x2, y2), 10, (255, 0, 255), -1)
                    # print(x1,y1,x2,y2)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # elif bad_lines is not None :
    #     bad_grad = []
    #     for line in bad_lines:
    #         for x1, y1, x2, y2 in line:
    #             gard = max(y2, y1) - min(y2, y1) / max(x2, x1) - min(x2, x1)
    #             bad_grad.append(gard)
    #
    #     val, count = np.unique(np.array(bad_grad), return_counts=True)
    #     bad_parallel = val[count >= 2]
    #     for line in bad_lines:
    #         for x1, y1, x2, y2 in line:
    #             gard = max(y2, y1) - min(y2, y1) / max(x2, x1) - min(x2, x1)
    #             if gard in bad_parallel:
    #                 frame = cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
    #                 frame = cv2.circle(frame, (x2, y2), 10, (255, 0, 255), -1)
    #                 # print(x1,y1,x2,y2)
    #                 cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if frame_idx==1:
        prev_frame = gray_blur
    if frame_idx>2:
        pprev_frame = prev_frame
        prev_frame = gray_blur

    cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask)
    # Displaying image in gray_scale
    cv2.imshow("Gray Frame", gray)

    # Displaying the difference in currentframe to
    # the staticframe(very first_frame)
    # cv2.imshow("Difference Frame", diff_frame)

    # Displaying the black and white image in which if
    # intensity difference greater than 30 it will appear white
    cv2.imshow("diff threshold Frame", diff_frame)
    # cv2.imshow("b Frame", b)
    cv2.imshow("c Frame", canny)
    # cv2.imshow("d Frame", d)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    frame_idx+=1

#
# import pdb;pdb.set_trace()
# if conf > 0.75:
#     points_list.append([pt_x, pt_y])
# prev_pt = None
# pprev_pt = None
# prev_i = None
# pprev_i = None
# save_p_i = 0
# cos = 0
# y_point = 0
# for p_i, pt in enumerate(points_list):
#     # frame = cv2.circle(frame, (int(pt[1]), int(pt[0])), circle_size, (255,0,0), -1)
#     if p_i >= 2:
#         # if euclidean_distance(prev_pt,pt)< int(frame.shape[0]//3):
#         prev_mv = [p1 - p2 for p1, p2 in zip(prev_pt, pt)]
#         pprev_mv = [p1 - p2 for p1, p2 in zip(pprev_pt, prev_pt)]
#         if prev_mv[0] != 0 and prev_mv[1] != 0 and pprev_mv[0] != 0 and pprev_mv[1] != 0:
#             cos = cos_sim(pprev_mv, prev_mv)
#             if cos >= -0.5 and cos <= 1:
#                 frame = cv2.line(frame, (int(pprev_pt[1]), int(pprev_pt[0])), (int(prev_pt[1]), int(prev_pt[0])),
#                                  (0, 0, 255), 3)
#                 save_p_i = p_i
#                 y_point = prev_pt[0]
#         else:
#             pass
#         if prev_i != p_i and prev_pt[0] != pt[0] and prev_pt[1] != pt[1]:
#             if cos >= -0.5 and cos <= 1:
#                 pprev_pt = prev_pt
#                 prev_pt = pt
#                 pprev_i = prev_i
#                 prev_i = p_i
#     elif p_i == 0:
#         pprev_pt = pt
#         pprev_i = p_i
#     elif p_i == 1:
#         prev_pt = pt
#         prev_i = p_i