import cv2
import os
import numpy as np
import mediapipe as mp

# MediaPipe hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) # <<< 여기가 수정된 부분입니다. 괄호를 여기서 닫아줍니다.
mp_drawing = mp.solutions.drawing_utils

# 데이터 저장을 위한 설정
DATA_PATH = os.path.join('MP_Data')  # 데이터 저장 폴더
actions = np.array(['rock', 'paper', 'scissors'])  # 동작 종류
no_sequences = 30  # 각 동작마다 수집할 영상(시퀀스) 개수
sequence_length = 30  # 1개 영상의 길이 (프레임 수)

# 데이터 수집을 위한 폴더 생성
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 데이터 수집을 위한 변수 설정
sequence = 0
action_idx = 0
frame_num = 0

print("Starting data collection...")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 화면에 현재 수집 상태 표시
    cv2.putText(image, 'Collecting frames for {} - Video Number {}'.format(
        actions[action_idx] if 'action_idx' in locals() else '...', sequence), 
        (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('Data Collection', image)

    # 키보드 입력 대기
    key = cv2.waitKey(10) # 키 입력 대기 시간을 10ms로 늘려 안정성 확보

    # ESC 키로 종료
    if key == 27:
        break

    # 데이터 수집 로직 (한 프레임씩 30프레임 저장)
    if frame_num < sequence_length:
        if results.multi_hand_landmarks:
            landmarks = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
            
            # 현재 시퀀스 폴더에 프레임 데이터 저장
            npy_path = os.path.join(DATA_PATH, actions[action_idx], str(sequence), str(frame_num) + '.npy')
            np.save(npy_path, landmarks)
            frame_num += 1
    else:
        # 다음 시퀀스로 이동
        sequence += 1
        frame_num = 0
        if sequence == no_sequences:
            # 다음 동작으로 이동
            if action_idx < len(actions) - 1:
                action_idx += 1
                sequence = 0
            else:
                # 모든 데이터 수집 완료
                print("All data collected!")
                break
        
        # 다음 영상 수집을 위해 잠시 대기
        cv2.waitKey(2000)

cap.release()
cv2.destroyAllWindows()
