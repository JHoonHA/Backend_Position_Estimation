from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import time
 
app = FastAPI()

origins = [
    "http://YOUR_REMOTE_IP:8000",  # 필요하면 추가하셈
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solution.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


def calculate_angle(a, b, c):
    
    # 각 값을 받아 넘파이 배열로 변형
    a = np.array(a) # 첫번째
    b = np.array(b) # 두번째
    c = np.array(c) # 세번째 

    # 라디안을 계산하고 실제 각도로 변경한다.
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    # 180도가 넘으면 360에서 뺀 값을 계산한다.
    if angle >180.0:
        angle = 360-angle

    # 각도를 리턴한다.
    return angle

@app.post("/analyze")
async def analyze_pose(file: UploadFile = File(...)):
      # 이미지 읽어옴
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    status = 0

      # MediaPipe 자세 추정
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return JSONResponse(content ={"status":"자리비움","keypoints":[]})
    # 좌표 추출
    landmarks = results.pose_landmarks.landmark
    LEFT_SHOULDER = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    RIGHT_SHOULDER = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    NOSE = landmarks[mp_pose.PoseLandmark.NOSE]

    

    #좌표 변환   
    left_shoulder = (LEFT_SHOULDER.x, LEFT_SHOULDER.y)
    right_shoulder = (RIGHT_SHOULDER.x, RIGHT_SHOULDER.y)
    nose = (NOSE.x, NOSE.y)
    #어깨 기울기
    shoulder_slope = abs (LEFT_SHOULDER.y - RIGHT_SHOULDER.y)
    #머리위치 
    shoulder_center_y = (LEFT_SHOULDER.y + RIGHT_SHOULDER.y) / 2
    head_position = nose[1] - shoulder_center_y
    #자세판별
    if shoulder_slope < 0.05 and head_position > -0.05 and head_position < 0.1:
        status = "정자세"
    elif shoulder_slope >= 0.05:
        status = "기울어짐"
    elif head_position <= -0.1:
        status = "엎드림"
    else:
        status = "자리 비움"

    keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
    return JSONResponse(content={"status": status, "keypoints": keypoints})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)