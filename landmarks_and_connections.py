import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_V4L2)
# Initiate holistic model
with mp_holistic.Holistic(
min_detection_confidence=0.5,
min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        # show output frame
        ret, frame = cap.read()
        
        # Recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)        
        
        # Recolor image back to BGR for rendering with OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Drawing face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0),thickness=1, circle_radius=1),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 255, 163),thickness=1, circle_radius=1))
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 33),thickness=2, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 242),thickness=2, circle_radius=2))
        
        # 3. Left hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 33),thickness=2, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 242),thickness=2, circle_radius=2))
        
        #4. Pose Detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0),thickness=1, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 242),thickness=2, circle_radius=3))
        
        
        cv2.imshow('Holistic Model Detections', image)
        key = cv2.waitKey(10) & 0xFF 
        
        # if q key is pressed break the loop
        if key == ord('q'):
            break
    # clean up
    
cap.release()
cv2.destroyAllWindows()