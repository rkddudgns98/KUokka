import cv2, dlib
import numpy as np
import sys, os
from datetime import datetime

# margin
margin_x = 5
margin_y = 10

# direction and location
direction = 0   # 얼굴 방향 (왼쪽, 오른쪽)
location = 0    # 얼굴 위치 (위쪽, 아래쪽)
n_face = 0
y_face = 0

# resize scaler
scaler = 1

# initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

# load video
cap = cv2.VideoCapture(0)

face_roi = []
face_sizes = []
    
# loop
while True:
    # read frame buffer from video
    ret, img = cap.read()
    if not ret:
        break

    # resize frame
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    ori = img.copy()

    # find faces
    if len(face_roi) == 0:
        faces = detector(img, 1)
    else:
        roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
        # cv2.imshow('roi', roi_img)
        faces = detector(roi_img)


    # ------------------------------------------------------------------
    # ----------------------------no-face-------------------------------
    # ------------------------------------------------------------------
    # no faces
    if len(faces) == 0:
        print('no faces!')
    
        direction = 3
        location = 3

        # time
        now = datetime.now()
        n_face = int(now.second)

        if n_face < y_face:
            n_face += 60

        if n_face - y_face > 3:
            # find faces
            face_roi = []
            continue

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    # find facial landmarks
    for face in faces:
        if len(face_roi) == 0:
            dlib_shape = predictor(img, face)
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
        else:
            dlib_shape = predictor(roi_img, face)
            shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])

        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    
        # compute face center
        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

        # compute face boundaries
        min_coords = np.min(shape_2d, axis=0)
        max_coords = np.max(shape_2d, axis=0)

        # draw min, max coords
        cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        # compute face size
        face_size = max(max_coords - min_coords)
        face_sizes.append(face_size)
        if len(face_sizes) > 10:
            del face_sizes[0]
        mean_face_size = int(np.mean(face_sizes) * 1.8)

        # compute face roi
        face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
        face_roi = np.clip(face_roi, 0, 10000)

        now = datetime.now()
        y_face = int(now.second)

        # ------------------------------------------------------------------
        # ----------------------------face angle----------------------------
        # ------------------------------------------------------------------
            
        # eyes
        left_eye = [int((shape_2d[0][0]+shape_2d[1][0])/2), int((shape_2d[0][1]+shape_2d[1][1])/2)]
        right_eye = [int((shape_2d[2][0]+shape_2d[3][0])/2), int((shape_2d[2][1]+shape_2d[3][1])/2)]
        cv2.circle(img, center=tuple(left_eye), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, center=tuple(right_eye), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # nose
        nose = shape_2d[4]
        cv2.circle(img, center=tuple(nose), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                        
        # direction - left or right
        # 0 - center, 1 - left, 2 - right
        if (left_eye[0] - nose[0]) > (nose[0] - right_eye[0]):
            if (left_eye[0] - nose[0]) - (nose[0] - right_eye[0]) > margin_x:
                direction = 2   # right
            else:
                direction = 0   # center
        else:
            if (nose[0] - right_eye[0] - (left_eye[0] - nose[0])) > margin_x:
                direction = 1   # left   
            else:
                direction = 0   # center

        # center_screen, center_face
        height, width, channel = img.shape
        center_screen = [int(width/2), int(height/2)]
        center = [center_x, center_y]
        cv2.circle(img, center=tuple(center_screen), radius=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, center=tuple(center), radius=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # location - top or bottom
        # 0 - center, 1 - top, 2 - bottom
        if center[1] > height/2:
            if center[1] - height/2 > margin_y:
                location = 2   # bottom
            else:
                location = 0   # center
        else:
            if height/2 - center[1] > margin_y:
                location = 1   # top
            else:
                location = 0   # center 

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ------------------------------result------------------------------
    # ------------------------------------------------------------------
      
    # 각 경우에 맞게 모터 제어 필요
    os.system('cls')

    # time
    print(n_face - y_face)

    # direction
    if direction == 0 :
        print("direction : center")
    elif direction == 1 :
        print("direction : left") 
    elif direction == 2 :
        print("direction : right")
    elif direction == 3 :
        print("direction : no face")

    # location
    if location == 0 :
        print("location : center")
    elif location == 1 :
        print("location : top")  
    elif location == 2 :
        print("location : bottom") 
    elif location == 3 :
        print("location : no face")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    # visualize
    #cv2.imshow('original', ori)
    img = cv2.flip(img, 1)
    cv2.imshow('facial landmarks', img)

    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)