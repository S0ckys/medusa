import os, sys, time
import cv2
import mediapipe as mp
import pyautogui as pg
from eyetrax import GazeEstimator, run_9_point_calibration

mp_hands = mp.solutions.hands

class landmarker_and_result():
    def __init__(self):
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.createLandmarker()

def createLandmarker(self):
  # callback function
    def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result

    # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
    options = mp.tasks.vision.HandLandmarkerOptions( 
        base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), # path to model
        running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
        num_hands = 2, # track both hands
        min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
        min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
        min_tracking_confidence = 0.3, # lower than value to get predictions more often
        result_callback=update_result)
    
    # initialize landmarker
    self.landmarker = self.landmarker.create_from_options(options)



def hand_track(display=False, camera=1):
    #setup the model
    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(camera)                                              
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    if not cap.isOpened():
        print("Error: Could not open webcam."); return
    if display:
        cv2.namedWindow("Hand", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        if not ret:
            print("Error: Failed to read frames with cv2.")
            break
        
        #TODO: Hand Tracking
        res = hands.process(rgb_frame)

        if res.multi_hand_landmarks:
            x = res.multi_hand_landmarks[0]
            print(res.multi_hand_landmarks[0])

        #TODO: Eye Tracking
        # output = face_mesh.process(rgb_frame)
        # landmarks_points = output.multi_face_landmarks 
        # frame_h, frame_w, _ = frame.shape

        # if landmarks_points:
        #     landmarks = landmarks_points[0].landmark
        #     for landmark in landmarks[474:478]:
        #         x = int(landmark.x * frame_w)
        #         y = int(landmark.y * frame_h)
        #         z = landmark.z
        #         print (x, y, z)
        #         cv2.circle(frame, (x,y), 3, (0, 255, 0))    

        # features, blink = estimator.extract_features(frame)

        # if features is not None and not blink:
        #     x, y = estimator.predict([features])[0]
        #     print(f"Gaze: ({x:.0f}, {y:.0f})")
        # elif features is not None and blink:
        #     print("Blink!")

        #TODO: hand tracking: figure out which hand movements and how to SOTA detection (it should feel as good as apple)

        if display:
            cv2.imshow('Eye Controlled Mouse', frame)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = 1
    estimator = GazeEstimator()
    estimator.load_model("gaze_model.pkl")
    if not cv2.VideoCapture(cam).isOpened():
        cam = 0

    hand_track(display=True, camera=cam)                    # set to true to show image for debugging
    print("hi")