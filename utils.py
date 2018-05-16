import cv2
import math

teacher_pos = (254, 101)

def find_contours(mask, frame):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]

    centers = []
    for c in cnts:
        # ((x, y), radius) = cv2.minEnclosingCircle(c)
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        centers.append(center)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
    return centers


def dist_to_teacher(p):
    return math.pow(p[0] - teacher_pos[0], 2) + math.pow(p[1] - teacher_pos[1], 2) 

def find_teacher(centers):
    global teacher_pos
    closest = min(centers, key=dist_to_teacher)
    d = dist_to_teacher(closest)
    # print ("closest: ", closest, "dist: ", d)
    if d < 500:
        teacher_pos = closest
    return teacher_pos
