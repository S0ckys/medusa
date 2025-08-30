# import numpy as np
# from eyetrax import GazeEstimator, run_9_point_calibration
# from mediapipe.framework.formats import landmark_pb2
import cv2
import copy, sys, os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions as mp_solutions

from handmark_activation import *

from collections import deque, Counter

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

mp_hands = mp.solutions.hands

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

def hand_track(display=False, camera=0):

    #setup the model
    cap = cv2.VideoCapture(camera) 
    args = get_args()

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    cap_device = args.device
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    model = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=args.num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    if not cap.isOpened():
        print("Error: Could not open webcam."); return
    if display:
        cv2.namedWindow("Hand", cv2.WINDOW_NORMAL)
        
    #Main Loop
    while True:

        #for modes! very smart implementation here: https://github.com/kinivi/hand-gesture-recognition-mediapipe?tab=readme-ov-file
        mode = 0
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        #start the camera
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        if not ret:
            print("Error: Failed to read frames with cv2.")
            break
        
        #TODO: Hand Tracking
        frame_analysis = model.process(frame)

        if frame_analysis.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(frame_analysis.multi_hand_landmarks,
                      frame_analysis.multi_handedness):

                #draw da hands
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                print(hand_sign_id)

                #TODO: !!!!!!!!!!!!!! This is how we're going to lock the FUCK into this
                if hand_sign_id != 0:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()
                
                print(finger_gesture_history)

        else:
            point_history.append([0, 0])

        if display:
            cv2.imshow('Eye Controlled Mouse', debug_image)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = 1
    if not cv2.VideoCapture(cam).isOpened():
        cam = 0
    hand_track(display=True, camera=0) #conver to cam                           # set to true to show image for debugging
