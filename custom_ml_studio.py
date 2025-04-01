import os
import json
import logging
import torch
import sys
from flask import Flask, request, jsonify
from label_studio_ml.model import LabelStudioMLBase
from yolov5 import detect
from PIL import Image
import numpy as np
import cv2

logging.basicConfig(level=logging.DEBUG if '--debug' in sys.argv else logging.INFO)
logger = logging.getLogger(__name__)

LABEL_STUDIO_DATA_DIR = os.getenv('LABEL_STUDIO_DATA_DIR', os.path.expanduser('~/.local/share/label-studio'))

class CustomMLStudio(LabelStudioMLBase):
    def __init__(self, model_path='yolov5s.pt', **kwargs):
        super(CustomMLStudio, self).__init__(**kwargs)
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.from_name, self.to_name, self.labels = self.parse_label_config(kwargs.get('label_config'))
        self.load_model()

    def parse_label_config(self, label_config):
        if not label_config:
            logger.warning("Label config is not provided")
            return "label", "image", []
        
        import xml.etree.ElementTree as ET
        root = ET.fromstring(label_config)
        
        # Find RectangleLabels configuration
        rectangle_labels = root.find(".//RectangleLabels")
        if (rectangle_labels is None):
            return "label", "image", []
            
        from_name = rectangle_labels.get('name', 'label')
        to_name = rectangle_labels.get('toName', 'image')
        labels = [label.get('value') for label in rectangle_labels.findall('Label')]
        
        return from_name, to_name, labels

    def load_model(self):
        """
        Loads the YOLOv5 model based on the configuration.
        """
        logger.info("Loading YOLOv5 model from %s", self.model_path)
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=False)
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            self.model = None

    def predict(self, tasks, **kwargs):
        """
        Generates predictions for a batch of tasks using YOLOv5.
        """
        logger.info("Predicting for %d tasks", len(tasks))
        predictions = []

        for task in tasks:
            logger.debug("Processing task: %s", task)
            image_path = task['data'].get('image', '')
            if not image_path:
                logger.warning("No image found in task data")
                continue

            # Remove 'data/upload' prefix if it exists in the image_path
            image_path = image_path.replace('data/upload/', '', 1)
            full_image_path = os.path.join(LABEL_STUDIO_DATA_DIR, 'media', 'upload', image_path.lstrip('/'))
            
            try:
                with Image.open(full_image_path) as img:
                    img_width, img_height = img.size
                
                # Run YOLOv5 detection
                model_output = self.model(full_image_path)
                detections = model_output.xyxy[0].cpu().numpy()

                task_prediction = {
                    'id': task['id'],  # Include task ID
                    'model_version': 'YOLOv5-v0.0.1',
                    'score': 0.0,
                    'result': []
                }
                
                for det in detections:
                    x_min, y_min, x_max, y_max, conf, cls = det
                    class_name = self.model.names[int(cls)]
                    
                    # Skip if class not in allowed labels
                    if self.labels and class_name not in self.labels:
                        continue
                    
                    # Convert coordinates to percentages
                    x_perc = (x_min / img_width) * 100
                    y_perc = (y_min / img_height) * 100
                    width_perc = ((x_max - x_min) / img_width) * 100
                    height_perc = ((y_max - y_min) / img_height) * 100

                    # Update max confidence score
                    task_prediction['score'] = max(task_prediction['score'], float(conf))

                    # Create prediction entry
                    task_prediction['result'].append({
                        'from_name': self.from_name,
                        'to_name': self.to_name, 
                        'type': 'rectanglelabels',
                        'score': float(conf),
                        'value': {
                            'x': float(x_perc),
                            'y': float(y_perc),
                            'width': float(width_perc),
                            'height': float(height_perc),
                            'rectanglelabels': [class_name]
                        }
                    })

                predictions.append(task_prediction)
                logger.debug("Prediction for task %s: %s", task['id'], task_prediction)

            except Exception as e:
                logger.error(f"Error processing task {task['id']}: {str(e)}")
                continue

        response = {'predictions': predictions} if predictions else {'predictions': []}
        logger.debug("Returning response: %s", response)
        return response

    def fit(self, completions, workdir=None, **kwargs):
        """
        Placeholder for training logic. Currently, YOLOv5 training is not implemented here.

        Args:
            completions (list): List of labeled data from Label Studio.
            workdir (str): Directory for saving model checkpoints or logs.

        Returns:
            dict: Information about the training status.
        """
        logger.info("Training is not implemented for YOLOv5 in this example")

        training_status = {
            'detail': "Training is not implemented",
            'workdir': workdir
        }

        return training_status


if __name__ == "__main__":
    import argparse
    from label_studio_ml.api import init_app

    parser = argparse.ArgumentParser(description="Run the Custom YOLOv5 backend for Label Studio")
    parser.add_argument("--port", type=int, default=5050, help="Port to run the server on (default: 5050)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--model-path", type=str, default="yolov5s.pt", help="Path to the YOLOv5 model")
    parser.add_argument("--debug", action="store_true", help="Run the server in debug mode")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Initialize the Flask app
    app = Flask(__name__)

    # Initialize the CustomMLStudio instance
    custom_ml_studio = CustomMLStudio(model_path=args.model_path)

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            return jsonify({'status': 'OK', 'message': 'POST request received'})
        return "Custom YOLOv5 backend for Label Studio is running."

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'UP'})

    @app.route('/setup', methods=['GET', 'POST'])
    def setup():
        return jsonify({'status': 'OK'})

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Check if files were uploaded
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
                
            image_file = request.files['image']
            
            try:
                # Read image directly from memory instead of saving to disk
                image_array = np.frombuffer(image_file.read(), np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    return jsonify({'error': 'Failed to decode image'}), 400
                
                # Run prediction using the model
                results = custom_ml_studio.model(img)
                
                # Convert results to the expected format
                detections = []
                for det in results.xyxy[0].cpu().numpy():
                    x1, y1, x2, y2, conf, cls = det
                    class_name = custom_ml_studio.model.names[int(cls)]
                    
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'label': class_name,
                        'score': float(conf)
                    }
                    detections.append(detection)
                
                return jsonify({'predictions': detections})
                
            except Exception as e:
                logger.error(f"Image processing error: {str(e)}")
                return jsonify({'error': f'Image processing error: {str(e)}'}), 500
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/webhook', methods=['POST'])
    def webhook():
        event = request.json
        logger.debug("Received event: %s", event)
        if event['action'] == 'predict':
            if 'tasks' not in event:
                logger.error("No 'tasks' key in event")
                return jsonify({'status': 'error', 'message': "No 'tasks' key in event"}), 400
            
            tasks = event['tasks']
            result = custom_ml_studio.predict(tasks)
            
            # Format response for Label Studio
            response = {
                'results': result['predictions'] if isinstance(result, dict) and 'predictions' in result else []
            }
            
            logger.debug("Final response: %s", response)
            return jsonify(response)  # Return dict with 'results' key
        else:
            return jsonify({'status': 'Unknown event'})

    @app.route('/image_exists', methods=['POST'])
    def image_exists():
        data = request.json
        image_path = data.get('image_path', '')
        # Remove 'data/upload' prefix if it exists in the image_path
        image_path = image_path.replace('data/upload/', '', 1)
        full_image_path = os.path.join(LABEL_STUDIO_DATA_DIR, 'media', 'upload', image_path.lstrip('/'))
        logger.debug("Checking if image exists: %s", full_image_path)
        if os.path.exists(full_image_path):
            return jsonify({'exists': True})
        else:
            return jsonify({'exists': False, 'message': 'Image path does not exist'}), 404

    app.run(host=args.host, port=args.port, debug=args.debug)

