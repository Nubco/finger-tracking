import cv2
import mediapipe as mp

#mediapipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,          #maximum number of hands to detect
    min_detection_confidence=0.5, #minimum confidence value for hand detection to be considered successful
    min_tracking_confidence=0.5 #minimum confidence value for the hand landmarks to be considered tracked successfully
)
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.isOpened(0)

while cap.isOpend():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image =cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the
    image.flags.writeable = False
   # process the image and detect the hands
    results = hands.process(image)
   
   
   
    # if hands are detected draw the landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            #result video
            cv2.imshow('MediaPipe Hands', image)

            # 'q' shut down
            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()