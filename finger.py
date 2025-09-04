import cv2
import mediapipe as mp

# MediaPipe hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,               # 감지할 손의 최대 개수
    min_detection_confidence=0.5,  # 탐지 신뢰도
    min_tracking_confidence=0.5)   # 추적 신뢰도
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라 프레임을 가져올 수 없습니다.")
        continue

    # 이미지를 좌우 반전하고, BGR에서 RGB로 변환
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # 성능 향상을 위해 이미지를 읽기 전용으로 설정
    image.flags.writeable = False
    
    # 이미지 처리하여 손 감지
    results = hands.process(image)

    # 다시 그림을 그리기 위해 이미지를 쓰기 가능으로 설정
    image.flags.writeable = True
    
    # RGB 이미지를 다시 BGR로 변환 (OpenCV 표시용)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 손이 감지된 경우 랜드마크 그리기
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
    
    # 결과 영상 출력 (루프의 메인 레벨로 이동)
    cv2.imshow('MediaPipe Hands', image)

    # 'q' 키를 누르면 종료 (ESC 키 코드 27 대신 'q' 사용)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
