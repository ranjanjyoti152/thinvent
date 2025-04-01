from flask import Flask, request, Response
import requests
import cv2
import numpy as np
import logging
import os
import random
from collections import defaultdict
import time
import threading

app = Flask(__name__)

# Configuration
AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://192.168.100.125:5053")
AI_TIMEOUT = float(os.getenv("AI_TIMEOUT", 0.5))
COUNT_UPDATE_INTERVAL = float(os.getenv("COUNT_UPDATE_INTERVAL", 1.0))
MAX_RECONNECT_ATTEMPTS = int(os.getenv("MAX_RECONNECT_ATTEMPTS", 3))
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS  # ~0.0333 seconds
CONFIDENCE_THRESHOLD = 0.3  # Count objects with confidence > 30%

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Color mapping
PRESET_COLORS = {
    'person': (0, 255, 0),
    'car': (0, 0, 255),
    'truck': (255, 0, 0),
    'dog': (255, 255, 0),
    'cat': (255, 0, 255)
}
COLOR_MAP = {}

def get_color(label):
    if label in PRESET_COLORS:
        return PRESET_COLORS[label]
    if label not in COLOR_MAP:
        COLOR_MAP[label] = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255)
        )
    return COLOR_MAP[label]

@app.route('/')
def home():
    return "NVR App is running"

@app.route('/health', methods=['GET'])
def health():
    try:
        response = requests.get(f"{AI_SERVER_URL}/health", timeout=AI_TIMEOUT)
        return response.json(), response.status_code
    except requests.RequestException as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e)}, 503

@app.route('/video_feed')
def video_feed():
    rtsp_url = request.args.get('rtsp_url')
    if not rtsp_url or not isinstance(rtsp_url, str):
        return "Error: RTSP URL must be a valid string", 400
    return Response(
        generate_frames(rtsp_url),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def generate_frames(rtsp_url):
    # Stream-specific variables
    object_counts = defaultdict(int)
    last_count_update = time.time()
    error_count = 0
    reconnect_attempts = 0

    # Initialize video capture
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.error(f"Failed to open RTSP stream: {rtsp_url}")
        return

    # Optimize capture settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|dummy=1'

    # Thread-safe frame access
    frame_lock = threading.Lock()
    last_frame = None

    def update_frame():
        nonlocal cap, error_count, reconnect_attempts, last_frame
        success, frame = cap.read()
        with frame_lock:
            if success and frame is not None and isinstance(frame, np.ndarray):
                last_frame = frame
                error_count = 0
            else:
                error_count += 1
                last_frame = None
        return success

    # Background frame capture thread
    def capture_thread():
        nonlocal reconnect_attempts
        while reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
            if not update_frame():
                logger.warning(f"Frame read failed. Error count: {error_count}")
                if error_count >= 5:
                    logger.info("Reconnecting to RTSP stream...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(rtsp_url)
                    if not cap.isOpened():
                        reconnect_attempts += 1
                        logger.error(f"Reconnect attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS} failed")
                        if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                            logger.error("Max reconnect attempts reached.")
                            break
                    else:
                        reconnect_attempts = 0
            time.sleep(FRAME_INTERVAL / 2)

    capture_thread = threading.Thread(target=capture_thread, daemon=True)
    capture_thread.start()

    while reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
        try:
            start_time = time.time()

            # Get the latest frame
            with frame_lock:
                if last_frame is None:
                    time.sleep(FRAME_INTERVAL)
                    continue
                frame = last_frame.copy()

            # Encode frame
            ret, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                logger.error("Frame encoding failed, skipping...")
                time.sleep(FRAME_INTERVAL)
                continue

            files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}

            # Send to AI server
            try:
                response = requests.post(f"{AI_SERVER_URL}/predict", files=files, timeout=AI_TIMEOUT)
                response.raise_for_status()
                predictions = response.json()
            except (requests.Timeout, requests.RequestException) as e:
                logger.warning(f"AI server error: {str(e)}")
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                elapsed = time.time() - start_time
                if elapsed < FRAME_INTERVAL:
                    time.sleep(FRAME_INTERVAL - elapsed)
                continue

            # Update object counts (only for confidence > 0.3)
            current_time = time.time()
            if current_time - last_count_update >= COUNT_UPDATE_INTERVAL:
                object_counts.clear()
                last_count_update = current_time

            for pred in predictions.get('predictions', []):
                score = pred.get('score', 0)
                if score > CONFIDENCE_THRESHOLD:  # Only count if confidence > 0.3
                    label = pred.get('label', 'unknown')
                    object_counts[label] += 1

            # Draw counts
            y_offset = 30
            for label, count in object_counts.items():
                text = f"{label}: {count}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                y_offset += 30

            # Draw predictions (all detected objects, regardless of confidence)
            for pred in predictions.get('predictions', []):
                bbox = pred.get('bbox', [])
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    logger.warning(f"Invalid bbox format: {bbox}")
                    continue
                try:
                    if any(v is None or not isinstance(v, (int, float)) for v in bbox):
                        logger.warning(f"Invalid bbox values: {bbox}")
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    label = pred.get('label', 'unknown')
                    score = pred.get('score', 0)
                    color = get_color(label)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label}: {score:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 0, 0), -1)
                    cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing bbox {bbox}: {str(e)}")
                    continue

            # Encode and yield processed frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                logger.error("Failed to encode processed frame, skipping...")

            # Maintain 30 FPS
            elapsed = time.time() - start_time
            if elapsed < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - elapsed)

        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            time.sleep(FRAME_INTERVAL)
            continue

    cap.release()
    logger.info(f"Stream {rtsp_url} terminated.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
