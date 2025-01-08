from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp

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




@app.post("/analyze")
async def analyze_pose(file: UploadFile = File(...)):
      # 이미지 읽어옴
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

      # MediaPipe 자세 추정
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    keypoints = []
    if results.pose_landmarks:
        keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

    return JSONResponse(content={"keypoints": keypoints})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)