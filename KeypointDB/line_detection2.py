import cv2
import json
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy import interpolate

def getAngle(L1, L2):
    ang = math.degrees(math.atan2(L1[1], L1[0]) - math.atan2(L2[1], L2[0]))
    return ang + 360 if ang < 0 else ang


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

def make_points(image, average):
    slope, y_int = average
    y1 = image.shape[0]
    # y2 = int(y1 * (3/5))
    y2 = 0
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)

    return np.array([x1, y1, x2, y2])

def average(image, lines):
    left = []
    right = []
    for line in lines:
        # print(line)
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            left.append((slope, y_int))
    # right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    left_line = make_points(image, left_avg)
    x1,y1,x2,y2 = left_line

    image = cv2.circle(image, (x1,y1), 5, (0, 0, 255), -1)
    image = cv2.circle(image, (x2, y2), 5, (255, 0, 0), -1)
    # right_line = make_points(image, right_avg)
    return image, np.array([left_line])
def grey(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
def canny(image):
    edges = cv2.Canny(image,50,150)
    return edges

def region(image):
    height, width = image.shape
    triangle = np.array([
                       [(100, height), (475, 325), (width, height)]
                       ])

    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

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
def add_square_feature(X):
    X = np.concatenate([(X**2).reshape(-1,1), X], axis=1)
    return X

filename = 'test4_data'

json_filename = f'D:\\CCTV_Server\\golf\\output_json\\{filename}.json'
cap = cv2.VideoCapture(f'D:\\CCTV_Server\\golf\\korea_videos\\{filename}.mp4')
pts = read_keypoints(json_filename)
club_pts = pts[:,17,:]
left_wrist_pts = pts[:,9,:]
left_hip_pts = pts[:,11,:]
right_hip_pts = pts[:,12,:]
# Take first frame and find corners in it
ret, frame = cap.read()

height, width, _ = frame.shape
nof_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# Create a mask image for drawing purposes
mask = np.zeros_like(frame)

frame_idx = 0
is_club = False
# Assigning our static_back to None
static_back = None
club_p6 = None
club_p5 = None
club_p4 = None
club_p3 = None
club_p2 = None
club_p1 = None
club_p0 = None
left_wrist_p6 = None
left_hip_p6 = None
right_hip_p6 = None
center_p6 = None
# Time of movement
time = []
is_p1_p2 = False

# Initializing DataFrame, one column is start
# time and other column is end time

point_list = []
wrist_list = []
center_list = []
phase = 0
phase_str = ['Setup','Address','Toe-up','Mid-backswing','Top','Top2',
             'Mid-downswing','Mid-downswing2','Mid-downswing3',
             'Imapct','Mid-following-through','Finish','Finish2','Finish3']
polar_list = []
club_list = []
phase_change = False
linear_interpolate = False
X_list = []
y_list = []
x_total = []
y_total = []
while(1):
    ret,frame = cap.read()
    if not ret:
        break

    if club_pts[frame_idx, 2] > 0.7:
        is_club = True
    else:
        is_club = False

    if left_wrist_pts[frame_idx, 2] > 0.5:
        is_wrist = True
    else:
        is_wrist = False

    r = None

    if is_club and club_p0 is None:
        club_p0 = club_pts[frame_idx,:2]
        club_list.append([frame_idx,club_p0[0],club_p0[1]])
    if is_club and club_p1 is None:
        club_p1 = club_pts[frame_idx,:2]
        club_list.append([frame_idx,club_p1[0],club_p1[1]])
    if is_club and club_p2 is None:
        club_p2 = club_pts[frame_idx,:2]
        club_list.append([frame_idx,club_p2[0],club_p2[1]])
    if is_club and club_p3 is None:
        club_p3 = club_pts[frame_idx,:2]
        club_list.append([frame_idx,club_p3[0],club_p3[1]])
    if is_club and club_p4 is None:
        club_p4 = club_pts[frame_idx,:2]
        club_list.append([frame_idx,club_p4[0],club_p4[1]])
    if is_club and club_p5 is None:
        club_p5 = club_pts[frame_idx,:2]
        club_list.append([frame_idx,club_p5[0],club_p5[1]])

    elif is_club :
        club_p6 = club_pts[frame_idx,:2]
        club_list.append([frame_idx,club_p6[0],club_p6[1]])
        grad_p5_p6 = club_p6 - club_p5
        grad_p4_p5 = club_p5 - club_p4
        grad_p3_p4 = club_p4 - club_p3
        grad_p2_p3 = club_p3 - club_p2
        grad_p1_p2 = club_p2 - club_p1
        grad_p0_p1 = club_p1-club_p0
        grad_p0_p2 = club_p2 - club_p0
        grad_p2_p4 = club_p4 - club_p2
        grad_p4_p6 = club_p6 - club_p4
        grad_p0_p3 = club_p3 - club_p0
        grad_p3_p6 = club_p6 - club_p3
        # print(club_p2,club_p1,club_p0)
        # print(frame_idx,grad_p0_p1,grad_p1_p2)
        if grad_p0_p1[1]==0 and grad_p0_p1[0] ==0:
            # print('1')
            pass
        elif grad_p1_p2[1] == 0 and grad_p1_p2[0] == 0:
            # print('2')
            pass
        # cv2.putText(frame, 'p0:({},{}),p1:({},{}),p2:({},{}),p3:({},{}),p4:({},{}),p5:({},{}),p6:({},{})'
        #             .format(club_p0[0],club_p0[1],
        #                     club_p1[0],club_p1[1],
        #                     club_p2[0],club_p2[1],
        #                     club_p3[0],club_p3[1],
        #                     club_p4[0],club_p4[1],
        #                     club_p5[0], club_p5[1],
        #                     club_p6[0], club_p6[1]),
        #             (int(club_p1[0]),int(club_p1[1])), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255), 1)
        #
        # cv2.putText(frame, 'grad 1: %04f, grad 2: %04f'%(grad_p0_p1[1]/grad_p0_p1[0], grad_p1_p2[1]/grad_p1_p2[0]),
        #             (int(club_p1[0]),int(club_p1[1])), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255), 1)

        # if grad_p0_p3[1]>0 and  grad_p3_p6[1]<0:
        #     phase+=1 #1
        #     cv2.putText(frame,'TRANSITION ! ',(int(club_p6[0]), int(club_p6[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        # elif grad_p0_p1[1]>0 and grad_p2_p3[1]<0:
        #     phase+=1 #2
        # elif grad_p0_p1[0] > 0 and grad_p2_p3[0] < 0:
        #     phase+=1 #3

        point_list.append(club_p6)
        X_list.append(club_p6[0])
        y_list.append(club_p6[1])
        for point in point_list:
            frame = cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0,0,255), -1)
        if left_wrist_p6 is not None:
            cv2.line(frame, (int(left_wrist_p6[0]),int(left_wrist_p6[1])), (int(club_p6[0]), int(club_p6[1])), (0, 255, 0), 10)

        if phase_change:
            X = np.array(X_list,dtype=np.float64).reshape(len(X_list),1)
            y = np.array(y_list,dtype=np.float64)

            # if linear_interpolate:
            #     phi = np.linspace(0, 2. * np.pi, 40)
            #     r = 0.5 + np.cos(phi)  # polar coords
            #     x_1, y_1 = r * np.cos(phi), r * np.sin(phi)  # convert to cartesian
            #     import pdb;pdb.set_trace()
            #     tck, u = interpolate.splprep([x_1, y_1], s=0)
            #     new_points = interpolate.splev(u, tck)
            #     import matplotlib.pyplot as plt
            #
            #     fig, ax = plt.subplots()
            #     ax.plot(x_1, y_1, 'ro')
            #     ax.plot(new_points[0], new_points[1], 'r-')
            #     plt.show()
            #     linear_interpolate = False
            ###################### Polinomial RANSAC fiTTING ###############################################


            try:
                regressor = RANSACRegressor()
                print(X.shape,y.shape)
                regressor.fit(add_square_feature(X), y)
                inlier_mask = regressor.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)
            except:
                regressor = LinearRegression()
                regressor.fit(add_square_feature(X), y)

            # Predict data of estimated models
            line_X = np.arange(X.min(), X.max())[:, np.newaxis]
            line_y_ransac = regressor.predict(add_square_feature(line_X))
            phase_change = False

            line_X = line_X.reshape(len(line_X)).tolist()
            line_y_ransac = line_y_ransac.tolist()
            x_total.extend(line_X)
            y_total.extend(line_y_ransac)
            # print(phase,len(X_list))
            x_len = len(X_list)

            if phase>=0 and phase<=12:
                X_list = X_list[-int(x_len/5):]
                y_list = y_list[-int(x_len/5):]



            ################################################################################################
        club_p0 = club_p1
        club_p1 = club_p2
        club_p2 = club_p3
        club_p3 = club_p4
        club_p4 = club_p5
        club_p5 = club_p6
    if club_p6 is not None:
        cv2.putText(frame, 'phase : {}, count : {} , x: {} , y : {}'.format(phase_str[phase],frame_idx,club_p6[0], club_p6[1]), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'phase : {} , count : {} '.format(phase_str[phase], frame_idx), (50, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 2)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # contrast limit가 2이고 title의 size는 8X8
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = clahe.apply(gray)
    # polar
    if is_wrist:
        left_hip_p6 = left_hip_pts[frame_idx, :2]
        right_hip_p6 = right_hip_pts[frame_idx, :2]
        left_wrist_p6 = left_wrist_pts[frame_idx, :2]
        wrist_list.append(left_wrist_p6)
        for point in wrist_list:
            frame = cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        center_p6 = ((left_hip_p6 + right_hip_p6) / 2)
        center_list.append(center_p6)
        for point in center_list:
            frame = cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
        cv2.putText(frame, 'P  ', (int(center_p6[0]), int(center_p6[1]) - 20),
                    cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 2)

        cv2.line(frame, (int(center_p6[0]), int(center_p6[1])), (int(left_wrist_p6[0]), int(left_wrist_p6[1])),
                 (0, 255, 255), 10)
        cv2.line(frame, (int(center_p6[0]), int(center_p6[1])), (int(center_p6[0]), 0), (0, 255, 0), 5)
    ########################## 원점(hip 중심)을 기준으로 극좌표계를 구성한다 . ####################################
    # 양 hip 사이를 극 또는 원점으로 부른다.(center)
    # 데카르트 좌표계에서 x축 양의 방향을 극축으로 잡는다.
    # 점 P가 극과 떨어진 거리 OP=r 과 OP이 극축과 이루는 각 theta의 순서쌍 (r, theta)로 나타내고 이를 극좌표라고 한다.

        O = center_p6
        L2 = np.array([center_p6[0],0])-O
        L1 = left_wrist_p6 - O

        # print(L2,L1)
        theta_1 = getAngle(L2,L1)
        theta_1_int = int(theta_1)
        if phase==0 :
            if theta_1_int>=150 and theta_1_int<180:
                phase =2  # Address
                # phase_change = True
        if phase==1 :
            if theta_1_int>=60 and theta_1_int<90:
                phase =2  # Toe-up
                linear_interpolate = True
                phase_change = True
        if phase==2 :
            if theta_1_int>=30 and theta_1_int<60:
                phase =3  # Mid-backswing
                phase_change = True
        if phase ==3 :
            if theta_1_int>=15 and theta_1_int <30:
                phase =4  # Top
                phase_change = True
        if phase ==4 :
            # if club_p6 is not None:
            #     L3 = club_p6 - left_wrist_p6
            #     theta_2 = getAngle(L2, L3)
            #     theta_2_int = int(theta_2)
            # if np.dot(L3,L2)==0:
            if theta_1_int >= 0 and theta_1_int < 15:
                phase = 5 # top2
                phase_change = True
        if phase == 5 :
            if theta_1_int>=30 and theta_1_int<50:
                phase = 7  # Mid-downswing
                phase_change = True
        if phase ==6 :
            if theta_1_int>=70 and theta_1_int<90:
                phase = 7  # Mid-downswing2
                phase_change = True
        if phase ==7 :
            if theta_1_int>=90 and theta_1_int<120:
                phase = 8  # Mid-downswing3
                phase_change = True
        if phase == 8 :
            if L1[0]>0 :
                phase = 9  # Impact
                phase_change = True
        if phase == 9 :
            if theta_1_int>=300 and theta_1_int <330:
                phase = 10  # Mid-Following-through
                phase_change = True
        if phase == 10 :
            if theta_1_int>=330 and theta_1_int<360:
                phase =11  # Finish
                phase_change = True
        if phase == 11:
            if theta_1_int >= 0 and theta_1_int < 30:
                phase = 12 # Finish2
                phase_change = True
        if phase ==12:
            if frame_idx==nof_frames-5:
                phase = 13
                phase_change = True
        # theta = min(theta_1,math.pi-theta_1)
        # x = width - center_p6[0]
        r = np.linalg.norm(L1)

        polar_list.append([frame_idx,r,theta_1,left_wrist_p6[0],left_wrist_p6[1]])
        cv2.putText(frame, 'theta : {}  '.format(theta_1), (int(center_p6[0])+30, int(center_p6[1]) - 20),
                    cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 2)

    ###########################################################################################################
    for i, j in zip(x_total, y_total):
        frame = cv2.circle(frame, (int(i), int(j)), 5, (0, 255, 255), -1)




    # ----THE PREVIOUS ALGORITHM----#
    gaus = gauss(frame)
    # if static_back is None:
    #     static_back = gaus
    #     continue
    candidate_lines = None
    # diff_frame = cv2.absdiff(static_back, gaus)
    edges = cv2.Canny(gaus, 50, 150)
    # isolated = region(edges)
    # good_lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 40, np.array([]), minLineLength=150, maxLineGap=5)
    # # bad_lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 40, np.array([]), minLineLength=150, maxLineGap=5)
    # if good_lines is not None:
    #     # print(good_lines.shape)
    #     for line in good_lines:
    #         x1, y1, x2, y2 = line[0]
    #         # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
    #         candidate_lines = good_lines
    # # elif bad_lines is not None:
    # #     # print(bad_lines.shape)
    # #     for line in bad_lines:
    # #         x1, y1, x2, y2 = line[0]
    # #         cv2.line(frame, (x1, y1), (x2, y2), q(0, 255, 0), 10)
    # #         candidate_lines = bad_lines
    #
    # try:
    #     frame, averaged_lines = average(frame, candidate_lines)
    #     black_lines = display_lines(frame, averaged_lines)
    #     frame = cv2.addWeighted(frame, 0.8, black_lines, 1, 1)
    # except:
    #     pass
    cv2.imshow('frame', frame)
    cv2.imshow('edges', edges)
    # ----THE PREVIOUS ALGORITHM----#

    frame_idx+=1


    end_flag = False
    if cv2.waitKey(10) & 0xFF == 32:
        while (1):
            if cv2.waitKey(10) & 0xFF == 32: break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                end_flag=True
                break

    if cv2.waitKey(10) & 0xFF == ord('q') or end_flag:
        break

import csv
f = open('polar_.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
for i in range(len(polar_list)):
    wr.writerows([polar_list[i]])
f.close()
f = open('club_.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
for i in range(len(club_list)):
    wr.writerows([club_list[i]])
f.close()

cap.release()
cv2.destroyAllWindows()