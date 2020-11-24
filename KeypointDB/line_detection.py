import cv2
import json
import numpy as np
import pandas as pd


def ransac_polyfit(x, y, order=3, n=20, k=100, t=0.1, d=100, f=0.8):
    # Thanks https://en.wikipedia.org/wiki/Random_sample_consensus

    # n – minimum number of data points required to fit the model
    # k – maximum number of iterations allowed in the algorithm
    # t – threshold value to determine when a data point fits a model
    # d – number of close data points required to assert that a model fits well to data
    # f – fraction of close data points required

    besterr = np.inf
    bestfit = None
    for kk in range(k):
        maybeinliers = np.random.randint(len(x), size=n)
        maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
        alsoinliers = np.abs(np.polyval(maybemodel, x) - y) < t
        if sum(alsoinliers) > d and sum(alsoinliers) > len(x) * f:
            bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
            thiserr = np.sum(np.abs(np.polyval(bettermodel, x[alsoinliers]) - y[alsoinliers]))
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
    return bestfit

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


json_filename = 'D:\\CCTV_Server\\golf\\output_json\\out.json'
cap = cv2.VideoCapture('D:\\CCTV_Server\\golf\\korea_videos\\out.mp4')
pts = read_keypoints(json_filename)
club_pts = pts[:,17,:]


# Take first frame and find corners in it
ret, frame = cap.read()


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
# Time of movement
time = []
is_p1_p2 = False
# Initializing DataFrame, one column is start
# time and other column is end time
df = pd.DataFrame(columns = ["Start", "End"])

point_list = []
phase = 0
while(1):
    ret,frame = cap.read()
    if not ret:
        break

    if club_pts[frame_idx, 2] > 0.75:
        is_club = True
    else:
        is_club = False

    if is_club and club_p0 is None:
        club_p0 = club_pts[frame_idx,:2]
    if is_club and club_p1 is None:
        club_p1 = club_pts[frame_idx,:2]
    if is_club and club_p2 is None:
        club_p2 = club_pts[frame_idx,:2]
    if is_club and club_p3 is None:
        club_p3 = club_pts[frame_idx,:2]
    if is_club and club_p4 is None:
        club_p4 = club_pts[frame_idx,:2]
    if is_club and club_p5 is None:
        club_p5 = club_pts[frame_idx,:2]

    elif is_club :
        club_p6 = club_pts[frame_idx,:2]
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
        print(frame_idx,grad_p0_p1,grad_p1_p2)
        if grad_p0_p1[1]==0 and grad_p0_p1[0] ==0:
            print('1')
            pass
        elif grad_p1_p2[1] == 0 and grad_p1_p2[0] == 0:
            print('2')
            pass
        cv2.putText(frame, 'p0:({},{}),p1:({},{}),p2:({},{}),p3:({},{}),p4:({},{}),p5:({},{}),p6:({},{})'
                    .format(club_p0[0],club_p0[1],
                            club_p1[0],club_p1[1],
                            club_p2[0],club_p2[1],
                            club_p3[0],club_p3[1],
                            club_p4[0],club_p4[1],
                            club_p5[0], club_p5[1],
                            club_p6[0], club_p6[1]),
                    (int(club_p1[0]),int(club_p1[1])), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255), 1)

        cv2.putText(frame, 'grad 1: %04f, grad 2: %04f'%(grad_p0_p1[1]/grad_p0_p1[0], grad_p1_p2[1]/grad_p1_p2[0]),
                    (int(club_p1[0]),int(club_p1[1])), cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255), 1)

        if grad_p0_p3[1]>0 and  grad_p3_p6[1]<0:
            phase+=1 #1
            # cv2.putText(frame,'TRANSITION ! ',(int(club_p6[0]), int(club_p6[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        # elif grad_p0_p1[1]>0 and grad_p2_p3[1]<0:
        #     phase+=1 #2
        # elif grad_p0_p1[0] > 0 and grad_p2_p3[0] < 0:
        #     phase+=1 #3
        point_list.append(club_p2)
        # for point in point_list:
        #     frame = cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0,0,255), -1)

        club_p0 = club_p1
        club_p1 = club_p2
        club_p2 = club_p3
        club_p3 = club_p4
        club_p4 = club_p5
        club_p5 = club_p6
    # if club_p6 is not None:
    #     cv2.putText(frame, 'phase : {}, count : {} , x: {} , y : {}'.format(phase,frame_idx,club_p6[0], club_p6[1]), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2,
    #             (0, 0, 255), 2)
    # else:
    #     cv2.putText(frame, 'phase : {} , count : {} '.format(phase, frame_idx), (50, 50),
    #                 cv2.FONT_HERSHEY_PLAIN, 2,
    #                 (0, 0, 255), 2)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # contrast limit가 2이고 title의 size는 8X8
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = clahe.apply(gray)

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

cap.release()
cv2.destroyAllWindows()