import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import os

app = FastAPI()

# Allow CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model (nano version for speed)
# It will download automatically on first run
model = YOLO('yolov8n.pt') 

# RTSP Stream URL from user request
RTSP_URL = "rtsp://172.16.16.254:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"

def generate_frames():
    cap = cv2.VideoCapture(RTSP_URL)
    
    # Simple retry logic or fallback could be added here
    if not cap.isOpened():
        print(f"Error: Could not open video stream at {RTSP_URL}")
        # For testing purposes without the actual camera, you might want to fallback to 0 (webcam)
        # cap = cv2.VideoCapture(0) 

    while True:
        success, frame = cap.read()
        if not success:
            # If the stream drops, we try to reconnect
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        # Run YOLO inference on the frame
        results = model(frame)

        # Plot the results on the frame (draws bounding boxes)
        annotated_frame = results[0].plot()

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()

        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/")
def read_root():
    return {"message": "YOLO Object Detection API is running"}

@app.get("/video_feed")
def video_feed():
    """
    Returns a streaming response using MJPEG protocol.
    The frontend can display this directly in an <img> tag.
    """
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)

