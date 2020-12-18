# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import timeit
import numpy as np
from collections import OrderedDict
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])
def rotate_image(mat, angle ):

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

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the current
        # image to the ratio of distance between eyes in the
        # desired image
        dist = np.sqrt((dX * 2) + (dY * 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
        # return the aligned face
        return output,angle

def align_face(image):

  start = timeit.default_timer()
  

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('face_landmark/shape_predictor_68_face_landmarks.dat')
  fa = FaceAligner(predictor, desiredFaceWidth=256)
  # load the input image, resize it, and convert it to grayscale
  # img_path = '/Users/duc/Downloads/passport/yolov5/runs/detect/exp14/IMG_1319.JPG'
  # image = cv2.imread(img_path)
  # result = cv2.imread(img_path)
  result = image
  # print(result.shape)
  image = imutils.resize(image, width=800)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # show the original input image and detect faces in the grayscale
  # image
  # rects = detector(gray, 2)
  rects , score, _= detector.run(gray, 2)
  print(score)
  rotate_angle = 0
  final_angle = 0
  count = 0
  while len(score) == 0 or score[0] < 0.7 :
      rotate_angle = rotate_angle - 30
      final_angle = final_angle + rotate_angle
      gray = rotate_image(gray, rotate_angle)
      image = rotate_image(image, rotate_angle)
      rects , score, _= detector.run(gray, 1)
      count = count + 1
      print(score)
      if len(score) != 0:
        if score[0] > 0.7:
          # print(score)
          break
      if count == 15 :
        break

  for rect in rects:
    # extract the ROI of the original face, then align the face
    # using facial landmarks
    (x, y, w, h) = rect_to_bb(rect)
    print(x,y)
    faceAligned,angle = fa.align(image, gray, rect)
    # display the output images
    # cv2.imshow("Original", faceOrig)
    # cv2.imshow("Aligned", faceAligned)
    print(angle)
    result = rotate_image(result,angle + final_angle )
    cv2.imwrite('face_landmark/dest/1.png',faceAligned)
    cv2.imwrite('face_landmark/dest/2.png',result)

    stop = timeit.default_timer()

    print('Alignment Time: ', stop - start)

    return result

if __name__ == '__main__':
  image = cv2.imread('/Users/duc/Downloads/passport/yolov5/runs/detect/exp39/IMG_1420.jpg')
  
  align_face(image)
  