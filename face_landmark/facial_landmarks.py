# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import sys
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
import sys

import dlib

detector = dlib.get_frontal_face_detector()


# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.

img = cv2.imread(args["image"])
dets, scores, idx = detector.run(img, 1, -1)
# for i, d in enumerate(dets):
print("Detection {}, score: {}".format(dets, scores))

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)
def rotate_image(mat, angle):
  height, width = mat.shape[:2] # image shape has 3 dimensions
  image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

  # rotation calculates the cos and sin, taking absolutes of those.
  abs_cos = abs(rotation_mat[0,0])
  abs_sin = abs(rotation_mat[0,1])

  # find the new width and height bounds
  bound_w = int(height * abs_sin + width * abs_cos)
  bound_h = int(height * abs_cos + width * abs_sin)

  # subtract old image center (bringing image back to origo) and adding the new image center coordinates
  rotation_mat[0, 2] += bound_w/2 - image_center[0]
  rotation_mat[1, 2] += bound_h/2 - image_center[1]

  # rotate image with the new bounds and translated rotation matrix
  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat
if not rects:
  gray = rotate_image(gray,-45)
  image = rotate_image(image,-90)
  dets, rects, idx = detector(gray, 1)
  # if (len(sys.argv[1:]) > 0):
  #   img = dlib.load_rgb_image(sys.argv[1])
  #   dets, scores, idx = detector.run(image, 1, -1)
  for i, d in enumerate(dets):
    print("Detection {}, score: {}, face_type:{}".format(d, scores[i]))
if not rects:
  gray = rotate_image(gray,90)
  image = rotate_image(image,90)
  rects = detector(gray, 1)  
if not rects:
  gray = rotate_image(gray,-45)
  image = rotate_image(image,30)
  rects = detector(gray, 1)
# loop over the face detections
for (i, rect) in enumerate(rects):
  # determine the facial landmarks for the face region, then
  # convert the facial landmark (x, y)-coordinates to a NumPy
  # array
  shape = predictor(gray, rect)
  shape = face_utils.shape_to_np(shape)
  # convert dlib's rectangle to a OpenCV-style bounding box
  # [i.e., (x, y, w, h)], then draw the face bounding box
  (x, y, w, h) = face_utils.rect_to_bb(rect)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  # show the face number
  cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  # loop over the (x, y)-coordinates for the facial landmarks
  # and draw them on the image
  for (x, y) in shape:
    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
# show the output image with the face detections + facial landmarks
# cv2.imshow("Output", image)
# cv2.waitKey(0)
cv2.imwrite('/content/gdrive/MyDrive/face_landmark/dest/1.png',image)