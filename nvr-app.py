from flask import Flask, request, jsonify, Response
import requests
import cv2
import numpy as np
from PIL import Image
import io
import logging
import os
import random
from collections import defaultdict
import time

app = Flask(__name__)

AI_SERVER_URL = "http://localhost:5053"

# Add logging configuration at the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a color mapping for different object classes
COLOR_MAP = {}

# Predefined colors for common objects
PRESET_COLORS = {
    'person': (0, 255, 0),     # Green
    'car': (0, 0, 255),        # Red
    'truck': (255, 0, 0),      # Blue
    'dog': (255, 255, 0),      # Cyan
    'cat': (255, 0, 255)       # Magenta
}

object_counts = defaultdict(int)
last_count_update = time.time()
COUNT_UPDATE_INTERVAL = 1  # Update counts every second

def get_color(label):
    if label in PRESET_COLORS:
        return PRESET_COLORS[label]
    if label not in COLOR_MAP:
        color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255)
        )
        COLOR_MAP[label] = color
    return COLOR_MAP[label]

@app.route('/')
def home():
    return "NVR App is running"

@app.route('/health', methods=['GET'])
def health():
    response = requests.get(f"{AI_SERVER_URL}/health")
    return jsonify(response.json())

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if request.method == 'GET':
        response = requests.get(f"{AI_SERVER_URL}/setup")
    else:
        response = requests.post(f"{AI_SERVER_URL}/setup", json=request.json)
    return jsonify(response.json())

@app.route('/predict', methods=['POST'])
def predict():
    response = requests.post(f"{AI_SERVER_URL}/predict", json=request.json)
    return jsonify(response.json())

@app.route('/webhook', methods=['POST'])
def webhook():
    response = requests.post(f"{AI_SERVER_URL}/webhook", json=request.json)
    return jsonify(response.json())

@app.route('/image_exists', methods=['POST'])
def image_exists():
    response = requests.post(f"{AI_SERVER_URL}/image_exists", json=request.json)
    return jsonify(response.json())

def create_gstreamer_pipeline(rtsp_url):
    return (
        f'rtspsrc location="{rtsp_url}" latency=0 buffer-size=1024 '
        'tcp-timeout=5000000 retry=1 ! '
        'rtph265depay ! h265parse ! '
        'avdec_h265 max-threads=4 ! '
        'videoconvert ! video/x-raw,format=BGR ! '
        'videoscale ! video/x-raw,width=1280,height=720 ! '
        'appsink drop=1 max-buffers=2 sync=false'
    )

def generate_frames(rtsp_url):
    global last_count_update
    
    frame_count = 0
    error_count = 0
    max_errors = 5
    
    # Initialize local object counts
    local_object_counts = defaultdict(int)
    
    try:
        # Try GStreamer pipeline first
        pipeline = create_gstreamer_pipeline(rtsp_url)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            # Fall back to default FFMPEG if GStreamer fails
            logger.info("Falling back to FFMPEG capture")
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'
                'max_delay;30000|'
                'reorder_queue_size;1000|'
                'buffer_size;512000'
            )
            
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduced buffer size
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H265'))
            cap.set(cv2.CAP_PROP_FPS, 30)  # Increased FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception as e:
        logger.error(f"Failed to initialize capture: {str(e)}")
        return

    while True:
        try:
            success, frame = cap.read()
            
            if not success:
                error_count += 1
                logger.warning(f"Failed to read frame. Error count: {error_count}")
                
                if error_count >= max_errors:
                    logger.info("Attempting to reconnect to RTSP stream...")
                    cap.release()
                    time.sleep(1)  # Add delay before reconnecting
                    cap = cv2.VideoCapture(rtsp_url)
                    if not cap.isOpened():
                        logger.error("Failed to reconnect to stream")
                        time.sleep(2)  # Wait longer before next attempt
                    error_count = 0
                continue
            
            # Process every 4th frame instead of 3rd
            if frame_count % 4 != 0:
                frame_count += 1
                continue
                
            frame_count += 1
            error_count = 0

            # Reduce frame size if needed
            if frame.shape[1] > 1280:  # If width is larger than 1280
                scale = 1280 / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Convert frame to bytes
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Create files dictionary with image bytes
            files = {
                'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')
            }
            
            # Send frame to AI server for prediction with timeout
            response = requests.post(
                f"{AI_SERVER_URL}/predict", 
                files=files, 
                timeout=1.0  # 1 second timeout
            )
            
            if response.status_code == 200:
                predictions = response.json()
                current_time = time.time()
                
                # Reset counts periodically
                if current_time - last_count_update >= COUNT_UPDATE_INTERVAL:
                    local_object_counts.clear()
                    last_count_update = current_time
                
                # Count objects
                for pred in predictions.get('predictions', []):
                    label = pred.get('label', 'unknown')
                    local_object_counts[label] += 1
                
                # Draw object counts
                y_offset = 30
                for label, count in local_object_counts.items():
                    text = f"{label}: {count}"
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 255, 255), 2)
                    y_offset += 30
                
                # Draw predictions on frame
                for pred in predictions.get('predictions', []):
                    bbox = pred.get('bbox', [])
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        label = pred.get('label', 'unknown')
                        score = pred.get('score', 0)
                        
                        color = get_color(label)
                        
                        # Draw futuristic bounding box
                        # Main box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Corner lines
                        corner_length = 20
                        # Top-left
                        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, 3)
                        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, 3)
                        # Top-right
                        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, 3)
                        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, 3)
                        # Bottom-left
                        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, 3)
                        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, 3)
                        # Bottom-right
                        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, 3)
                        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, 3)
                        
                        # Add label with improved visibility
                        text = f"{label}: {score:.2f}"
                        font_scale = 0.7
                        thickness = 2
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        
                        # Get text size
                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        # Draw semi-transparent background
                        sub_img = frame[y1 - text_height - 10:y1, x1:x1 + text_width + 10]
                        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                        alpha = 0.4
                        frame[y1 - text_height - 10:y1, x1:x1 + text_width + 10] = \
                            cv2.addWeighted(sub_img, alpha, black_rect, 1 - alpha, 0)
                        
                        # Draw text with outline
                        cv2.putText(frame, text, (x1 + 5, y1 - 5), font, font_scale,
                                   (0, 0, 0), thickness + 2)  # outline
                        cv2.putText(frame, text, (x1 + 5, y1 - 5), font, font_scale,
                                   (255, 255, 255), thickness)  # text

            # Convert frame to JPEG format for streaming
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except requests.exceptions.Timeout:
            logger.warning("AI server prediction timeout")
            continue
            
        except Exception as e:
            logger.error(f"Error in frame processing: {str(e)}")
            if frame is not None:  # Check if frame exists before yielding
                # Convert frame to JPEG format for streaming
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
    
    cap.release()

# Update the video_feed route to include error handling
@app.route('/video_feed')
def video_feed():
    rtsp_url = request.args.get('rtsp_url')
    if not rtsp_url:
        return "Error: No RTSP URL provided", 400
        
    return Response(
        generate_frames(rtsp_url),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
