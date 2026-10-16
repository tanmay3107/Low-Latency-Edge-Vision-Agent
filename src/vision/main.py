import cv2
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.vision.agent import EdgeVisionAgent

app = FastAPI(title="EdgeVision Security Agent API")
agent = EdgeVisionAgent()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Open default web camera (or replace with RTSP stream URL for IP cameras)
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for optimal edge throughput
            frame = cv2.resize(frame, (640, 480))

            # Run hybrid inference agent
            analysis = agent.process_frame(frame)

            # Encode frame to JPEG bytes for WebSocket transmission
            _, buffer = cv2.imencode('.jpg', frame, [cv2.imencode_param("jpeg_quality", 80)])
            
            # Send telemetry JSON and frame payload
            await websocket.send_json({
                "analysis": analysis,
                "frame_shape": frame.shape
            })
            
            # Control frame rate (~15 FPS)
            await asyncio.sleep(0.06)

    except WebSocketDisconnect:
        print("Client disconnected from video stream.")
    finally:
        cap.release()