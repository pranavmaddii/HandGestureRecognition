import cv2
import mediapipe as mp

mp_hand = mp.solutions.hands
hand_detector = mp_hand.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

window_name = 'Hand Gesture Recognition'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window

fullscreen = False  # Flag to track fullscreen mode

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hand.HAND_CONNECTIONS)

            landmarks_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_list.append([cx, cy])

            def is_finger_extended(finger_tip_id, finger_pip_id):
                return landmarks_list[finger_tip_id][1] < landmarks_list[finger_pip_id][1]

            thumb_extended = landmarks_list[4][0] > landmarks_list[3][0]
            index_extended = is_finger_extended(8, 6)
            middle_extended = is_finger_extended(12, 10)
            ring_extended = is_finger_extended(16, 14)
            pinky_extended = is_finger_extended(20, 18)

            if thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended:
                gesture = "Call Me!"
            elif thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
                gesture = "High Five!"
            elif middle_extended and not index_extended and not ring_extended and not pinky_extended:
                gesture = "Not Cool!"
            elif (landmarks_list[8][0] < landmarks_list[6][0] and
                  landmarks_list[12][0] > landmarks_list[10][0] and
                  landmarks_list[16][0] > landmarks_list[14][0] and
                  landmarks_list[20][0] > landmarks_list[18][0]):
                gesture = "To the Left!"
            elif (landmarks_list[8][0] > landmarks_list[6][0] and
                  landmarks_list[12][0] < landmarks_list[10][0] and
                  landmarks_list[16][0] < landmarks_list[14][0] and
                  landmarks_list[20][0] < landmarks_list[18][0]):
                gesture = "To the Right!"
            elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture = "1"
            elif index_extended and middle_extended and not ring_extended and not pinky_extended:
                gesture = "2"
            elif index_extended and middle_extended and ring_extended and not pinky_extended:
                gesture = "3"
            elif index_extended and middle_extended and ring_extended and pinky_extended:
                gesture = "4"
            elif thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
                gesture = "5"
            elif (index_extended and not middle_extended and not ring_extended and not pinky_extended and
                  hand_landmarks.landmark[0].x < hand_landmarks.landmark[5].x):
                gesture = "5"
            elif thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                gesture = "Thumbs Up!"
            elif index_extended and not middle_extended and not ring_extended and pinky_extended:
                gesture = "Rock 'n' Roll!"
            else:
                gesture = None

            if gesture:
                cv2.putText(frame, gesture, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(window_name, frame)

    # Check for key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):  # Toggle fullscreen mode when 'f' key is pressed
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            fullscreen = False
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            fullscreen = True

cap.release()
cv2.destroyAllWindows()
