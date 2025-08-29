# import numpy as np
# from eyetrax import GazeEstimator, run_9_point_calibration
# from mediapipe.framework.formats import landmark_pb2
import cv2

import mediapipe as mp
from mediapipe import solutions as mp_solutions
from handmark_activation import *

mp_hands = mp.solutions.hands

def hand_track(display=False, camera=1):
    #setup the model
    cap = cv2.VideoCapture(camera) 
    args = get_args()

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
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not ret:
            print("Error: Failed to read frames with cv2.")
            break
        
        #TODO: Hand Tracking
        frame_analysis = model.process(rgb_frame)
        if frame_analysis.multi_hand_landmarks is not None:
            for landmark in frame_analysis.multi_hand_landmarks:
                mp_solutions.drawing_utils.draw_landmarks(
                    frame, landmark, mp_hands.HAND_CONNECTIONS)
        

        #TODO: hand tracking: figure out which hand movements and how to SOTA detection (it should feel as good as apple)

        if display:
            cv2.imshow('Eye Controlled Mouse', frame)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = 1
    if not cv2.VideoCapture(cam).isOpened():
        cam = 0

    hand_track(display=True, camera=cam)                    # set to true to show image for debugging
    print("hi")