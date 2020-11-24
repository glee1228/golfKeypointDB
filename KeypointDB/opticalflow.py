import numpy as np
import cv2
import json

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

json_filename = 'D:\\CCTV_Server\\golf\\output_json\\test4_data.json'
cap = cv2.VideoCapture('D:\\CCTV_Server\\golf\\output_trajectory\\test4_data.mp4')
pts = read_keypoints(json_filename)
club_pts = pts[:,17,:]

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.01,
                       minDistance = 30,
                       blockSize = 14)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

club_p0 = club_pts[0,:]
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

frame_idx = 0
is_club = False
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
        frame = cv2.circle(frame, (int(club_p2[1]), int(club_p2[0])), 5, (255,0,0), -1)
        # if frame_idx >= 2:
        #     # if euclidean_distance(prev_pt,pt)< int(frame.shape[0]//3):
        #     prev_mv = [p1 - p2 for p1, p2 in zip(prev_pt, pt)]
        #     pprev_mv = [p1 - p2 for p1, p2 in zip(pprev_pt, prev_pt)]
        #     if prev_mv[0] != 0 and prev_mv[1] != 0 and pprev_mv[0] != 0 and pprev_mv[1] != 0:
        #         cos = cos_sim(pprev_mv, prev_mv)
        #         if cos >= -0.5 and cos <= 1:
        #             frame = cv2.line(frame, (int(pprev_pt[1]), int(pprev_pt[0])), (int(prev_pt[1]), int(prev_pt[0])),
        #                              (0, 0, 255), 3)
        #             save_p_i = p_i
        #             y_point = prev_pt[0]
        #     else:
        #         pass
        #     if prev_i != p_i and prev_pt[0] != pt[0] and prev_pt[1] != pt[1]:
        #         if cos >= -0.5 and cos <= 1:
        #             pprev_pt = prev_pt
        #             prev_pt = pt
        #             pprev_i = prev_i
        #             prev_i = p_i
        # elif p_i == 0:
        #     pprev_pt = pt
        #     pprev_i = p_i
        # elif p_i == 1:
        #     prev_pt = pt
        #     prev_i = p_i
        #





    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    mv = good_new - good_old
    norm_mv=np.linalg.norm(mv,axis=1)
    max_norm_mv = np.max(norm_mv)
    max_idx = np.where(norm_mv == max_norm_mv)

    max_new = p1[max_idx]
    max_old = p0[max_idx]
    # draw the tracks
    for i,(new,old) in enumerate(zip(max_new,max_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    cv2.imshow('mask',mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)


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