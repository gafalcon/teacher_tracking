# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import utils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="./class.mp4",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
maskLower = [160, 96, 50]
maskUpper = [180, 255, 255]

# pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])

def change_mask(mask_type, i):
    if mask_type == "upper":
        def f(x):
            maskUpper[i] = x
            print ("upper: ", maskUpper)
            print ("lower: ", maskLower)
    else:
        def f(x):
            maskLower[i] = x
            print ("upper: ", maskUpper)
            print ("lower: ", maskLower)
    return f


cv2.namedWindow("original")

# create trackbars for color change
low_vars = ["h", "s", "v"]
high_vars = ["H", "S", "V"]
for i,v in enumerate(low_vars):
    cv2.createTrackbar(v, "original", maskLower[i], 255, change_mask("lower", i))

for i,v in enumerate(high_vars):
    cv2.createTrackbar(v, "original", maskUpper[i], 255, change_mask("upper", i))
# keep looping
n_frame = 1
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, tuple(maskLower), tuple(maskUpper))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)


    centers = utils.find_contours(mask, frame)
    teacher_pos = utils.find_teacher(centers)
    cv2.circle(frame, teacher_pos, 25, (0, 0, 255), 1)
    cv2.imshow("original", frame)
    cv2.imshow("binary", mask)
    key = cv2.waitKey(1) & 0xFF
    print (n_frame)
    n_frame += 1
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.waitKey(0)

camera.release()
cv2.destroyAllWindows()

# 10460
