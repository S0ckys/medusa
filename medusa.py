import os, sys, time, cv2
import mediapipe as mp
import pyautogui as pg
import numpy as np
from eyetrax import GazeEstimator, run_9_point_calibration
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions as mp_solutions


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
         base_options = mp.tasks.BaseOptions(model_asset_path="Hand Landmarker Task - Google AI Guide.task"), # path to model
         running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
         num_hands = 2, # track both hands
         min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
         min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
         min_tracking_confidence = 0.3, # lower than value to get predictions more often
         result_callback=update_result)
      
      # initialize landmarker
      self.landmarker = self.landmarker.create_from_options(options)
   
   def detect_async(self, frame):
      # convert np frame to mp image
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      # detect landmarks
      self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

   def close(self):
      # close landmarker
      self.landmarker.close()

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
    try:
        if detection_result is None or not detection_result.hand_landmarks:
            return rgb_image
        else:
            hand_landmarks_list = detection_result.hand_landmarks
            annotated_image = np.copy(rgb_image)

            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

                # build landmark proto
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    ) for landmark in hand_landmarks
                ])

                # draw landmarks + connections
                mp_solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp_solutions.hands.HAND_CONNECTIONS,
                    mp_solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp_solutions.drawing_styles.get_default_hand_connections_style()
                )
            return annotated_image
    except Exception as e:
        print("Drawing error:", e)
        return rgb_image




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

    #intializing the landmarker_and_result class
    hand_landmarker = landmarker_and_result()


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
        hand_landmarker.detect_async(rgb_frame)
        frame = draw_landmarks_on_image(frame,hand_landmarker.result)

        # print(hand_landmarker.result)


        # res = hands.process(rgb_frame)

        # if res.multi_hand_landmarks:
        #     x = res.multi_hand_landmarks[0]
        #     print(type(res.multi_hand_landmarks[0]))

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
    hand_landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = 1
    # estimator = GazeEstimator()
    # estimator.load_model("gaze_model.pkl")
    if not cv2.VideoCapture(cam).isOpened():
        cam = 0

    hand_track(display=True, camera=cam)                    # set to true to show image for debugging
    print("hi")