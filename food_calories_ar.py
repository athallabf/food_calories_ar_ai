import cv2
import numpy as np
import time
import json
import os
import torch
from pathlib import Path
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Import YOLOv5 detection modules
from detect import run, DetectMultiBackend

class FoodCaloriesAR:
    def __init__(self, weights_path='best.pt', device=''):
        """Initialize the Food Calories AR system"""
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrate_camera()
        
        # Original initialization
        self.food_database = {}
        self.load_food_database()
        self.last_time = 0
        self.fps = 0
        self.detected_foods = []
        
        # 2D visualization settings
        self.info_panel_height = 80
        self.info_panel_width = 200
        self.panel_margin = 10
        
        # Initialize YOLOv5 model
        print(f"Loading model from {weights_path}...")
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.imgsz = (640, 640)  # inference size
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        print("Model loaded successfully!")
        
        # AR display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.text_color = (255, 255, 255)
        self.box_color = (0, 255, 0)
        self.thickness = 2
        
        # Demo mode for testing without model
        self.demo_mode = False  # Set to False since we have the model now
        
    def calibrate_camera(self):
        """Basic camera calibration"""
        # Get camera resolution
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Approximate camera matrix for AR visualization
        focal_length = width
        center = (width/2, height/2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float32)
        
        # Assume no lens distortion
        self.dist_coeffs = np.zeros((4,1))
        
    def load_food_database(self):
        """Load the food calorie database from JSON file"""
            
        # If you have a JSON file, you can load it like this:
        if os.path.exists("food_database.json"):
            with open("food_database.json", 'r') as f:
                self.food_database = json.load(f)
            
        print(f"Food database loaded with {len(self.food_database)} items")
            
    def get_food_info(self, food_name):
        """Get calorie information for a detected food item"""
        food_name = food_name.lower()
        if food_name in self.food_database:
            return self.food_database[food_name]
        else:
            return {"calories": "Unknown", "serving_size": "Unknown"}
            
    def calculate_fps(self):
        """Calculate the frames per second"""
        current_time = time.time()
        fps = 1 / (current_time - self.last_time) if self.last_time > 0 else 0
        self.last_time = current_time
        self.fps = 0.9 * self.fps + 0.1 * fps  # Running average
        return self.fps
        
    def detect_food(self, frame):
        """
        Detect food items using YOLOv5 model
        """
        if self.demo_mode:
            return super().detect_food(frame)
            
        # Preprocess image for YOLOv5
        img = cv2.resize(frame, self.imgsz)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]
            
        # Inference
        pred = self.model(img, augment=False, visualize=False)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
        
        detected = []
        
        # Process detections
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to frame size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                
                # Process detections
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_id = int(cls)
                    class_name = self.names[class_id]
                    confidence = float(conf)
                    
                    # Get food info from database
                    food_info = self.get_food_info(class_name)
                    calories = food_info.get('calories', 'Unknown')
                    
                    detected.append({
                        "name": class_name,
                        "calories": calories,
                        "confidence": confidence,
                        "box": (x1, y1, x2-x1, y2-y1)  # Convert to (x, y, w, h) format
                    })
                    
        self.detected_foods = detected
        return detected
        
    def draw_food_info(self, frame, food_items):
        """Draw the food information on the AR display"""
        for food in food_items:
            name = food["name"]
            calories = food["calories"]
            confidence = food.get("confidence", 0) * 100
            x, y, w, h = food["box"]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.box_color, 2)
            
            # Create info box for food details
            info_bg = np.zeros((self.info_panel_height, self.info_panel_width, 3), dtype=np.uint8)
            info_bg[:, :] = (0, 0, 150)  # Dark blue background
            
            # Get detailed info
            food_info = self.get_food_info(name.lower())
            serving = food_info.get("serving_size", "Unknown")
            
            # Draw text on info box
            cv2.putText(info_bg, f"{name}", (10, 20), self.font, self.font_scale, self.text_color, 1)
            cv2.putText(info_bg, f"Calories: {calories}", (10, 40), self.font, self.font_scale, self.text_color, 1)
            cv2.putText(info_bg, f"Per: {serving}", (10, 60), self.font, self.font_scale, self.text_color, 1)
            
            # Find best position for info box (avoid overlapping with food)
            if y > 100:  # If enough space above the food
                info_y = max(0, y - self.info_panel_height - self.panel_margin)
                info_x = max(0, x)
            else:  # Place below the food
                info_y = min(frame.shape[0] - self.info_panel_height, y + h + self.panel_margin)
                info_x = max(0, x)
                
            # Make sure info box fits in frame
            info_x = min(info_x, frame.shape[1] - self.info_panel_width)
            
            # Add info box to frame with semi-transparency
            alpha = 0.7
            roi = frame[info_y:info_y+self.info_panel_height, info_x:info_x+self.info_panel_width]
            frame[info_y:info_y+self.info_panel_height, info_x:info_x+self.info_panel_width] = cv2.addWeighted(roi, 1-alpha, info_bg, alpha, 0)
            
            # Draw connecting line from box to food
            cv2.line(frame, (x + w//2, y + h//2), (info_x + 100, info_y + 40), (255, 255, 0), 1)
            
            # Draw confidence gauge
            gauge_x = info_x + 150
            gauge_y = info_y + 70
            gauge_length = 40
            gauge_height = 5
            # Background (gray)
            cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_length, gauge_y + gauge_height), (100, 100, 100), -1)
            # Foreground (colored by confidence)
            conf_width = int(gauge_length * confidence / 100)
            conf_color = (0, 255, 0) if confidence > 70 else (0, 165, 255) if confidence > 50 else (0, 0, 255)
            cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + conf_width, gauge_y + gauge_height), conf_color, -1)
            
        return frame
    
    def process_frame(self, frame):
        """Process a frame with 2D visualization"""
        # Detect food items
        detections = self.detect_food(frame)
        
        # Draw information for each detection
        frame = self.draw_food_info(frame, detections)
        
        # Add AR interface elements
        frame = self.add_ar_interface(frame)
        
        return frame

    def add_ar_interface(self, frame):
        """Add AR interface elements to the frame"""
        h, w = frame.shape[:2]
        
        # Add top status bar
        status_bar_height = 40
        status_bar = np.zeros((status_bar_height, w, 3), dtype=np.uint8)
        status_bar[:, :] = (50, 50, 50)  # Dark gray
        
        # Add info to status bar
        cv2.putText(status_bar, "Food Calories AR", (10, 30), self.font, 1, (255, 255, 255), 2)
        cv2.putText(status_bar, f"FPS: {self.calculate_fps():.1f}", (w - 150, 30), self.font, 1, (255, 255, 255), 1)
        
        # Status indicator for model
        cv2.putText(status_bar, "MODEL: Demo" if self.demo_mode else "MODEL: Active", (w // 2 - 70, 30), 
                    self.font, 0.7, (0, 255, 255) if self.demo_mode else (0, 255, 0), 2)
        
        # Combine status bar with main frame
        frame_with_ui = np.vstack((status_bar, frame))
        
        # Add instructions at the bottom
        cv2.putText(frame_with_ui, "Press 'q' to quit, 'd' to toggle demo mode", 
                    (10, h + status_bar_height - 10), self.font, 0.5, (255, 255, 255), 1)
        
        return frame_with_ui
    
    def run(self):
        """Run the main AR loop"""
        if not self.cap.isOpened():
            print("Error: Cannot access webcam")
            return
            
        print("Food Calories AR System")
        print("Press 'q' to quit")
        
        cv2.namedWindow("Food Calories AR", cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Display result
            cv2.imshow("Food Calories AR", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        
    def integrate_model(self, model_path):
        """
        Method to integrate the food detection model once it's available
        This will replace the placeholder demo detection
        """
        # This will be implemented once the model is available
        print(f"Model integration will be implemented for: {model_path}")
        self.demo_mode = False

# Function to prepare for model integration
def prepare_model_integration(food_ar):
    """
    This function will be used to integrate the model once it's available
    It will override the detect_food method with actual model inference
    """
    # This is just a placeholder structure for when the model is ready
    def model_inference(frame):
        # Replace this with actual model code
        # model_result = food_model.predict(frame)
        # detected = process_model_output(model_result)
        detected = []  # Will be populated by actual model
        return detected
    
    # Replace the detect_food method with actual model inference
    # food_ar.detect_food = model_inference
    pass

if __name__ == "__main__":
    food_ar = FoodCaloriesAR()
    
    # Later, when model is ready:
    # prepare_model_integration(food_ar)
    
    # Run the AR system
    food_ar.run()