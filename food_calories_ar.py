import cv2
import numpy as np
import time
import json
import os

class FoodCaloriesAR:
    def __init__(self):
        """Initialize the Food Calories AR system"""
        self.cap = cv2.VideoCapture(0)
        self.food_database = {}
        self.load_food_database()
        self.last_time = 0
        self.fps = 0
        self.detected_foods = []  # Will be populated by the model later
        
        # Placeholder until real model is integrated
        self.placeholder_foods = [
            {"name": "Apple", "calories": 95, "confidence": 0.92},
            {"name": "Banana", "calories": 105, "confidence": 0.89},
            {"name": "Pizza Slice", "calories": 285, "confidence": 0.78}
        ]
        
        # AR display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.text_color = (255, 255, 255)
        self.box_color = (0, 255, 0)
        self.thickness = 2
        
        # Demo mode for testing without model
        self.demo_mode = True
        self.demo_counter = 0
        
    def load_food_database(self):
        """Load the food calorie database from JSON file"""
        try:
            # You can replace this with your own food database
            sample_db = {
                "apple": {"calories": 95, "serving_size": "1 medium (182g)"},
                "banana": {"calories": 105, "serving_size": "1 medium (118g)"},
                "orange": {"calories": 62, "serving_size": "1 medium (131g)"},
                "rice": {"calories": 206, "serving_size": "1 cup cooked (158g)"},
                "bread": {"calories": 75, "serving_size": "1 slice (30g)"},
                "chicken_breast": {"calories": 165, "serving_size": "100g cooked"},
                "egg": {"calories": 78, "serving_size": "1 large (50g)"},
                "milk": {"calories": 103, "serving_size": "1 cup (244g)"},
                "pizza_slice": {"calories": 285, "serving_size": "1 slice (107g)"},
                "french_fries": {"calories": 365, "serving_size": "medium serving (117g)"}
            }
            
            # If you have a JSON file, you can load it like this:
            # if os.path.exists("food_database.json"):
            #     with open("food_database.json", 'r') as f:
            #         self.food_database = json.load(f)
            # else:
            self.food_database = sample_db
                
            print(f"Food database loaded with {len(self.food_database)} items")
            
        except Exception as e:
            print(f"Error loading food database: {e}")
            self.food_database = {}
            
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
        Placeholder for food detection model
        This will be replaced with actual model inference once available
        """
        # This is just a placeholder that simulates food detection 
        # by randomly placing boxes - will be replaced by your model
        
        h, w = frame.shape[:2]
        detected = []
        
        if self.demo_mode:
            # For demo purposes, show different foods in different regions
            self.demo_counter += 1
            if self.demo_counter % 30 == 0:  # Change detection every ~1 second
                # Reset detections occasionally to simulate new detections
                detected = []
                
                # Randomly choose 1-2 food items
                num_detections = np.random.randint(1, 3)
                for _ in range(num_detections):
                    food = np.random.choice(self.placeholder_foods)
                    x = np.random.randint(50, w - 200)
                    y = np.random.randint(50, h - 200)
                    width = np.random.randint(150, 300)
                    height = np.random.randint(150, 300)
                    
                    detected.append({
                        "name": food["name"],
                        "calories": food["calories"],
                        "confidence": food["confidence"],
                        "box": (x, y, width, height)
                    })
                    
                self.detected_foods = detected
            else:
                # Keep the current detections
                detected = self.detected_foods
                
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
            info_bg = np.zeros((80, 200, 3), dtype=np.uint8)
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
                info_y = max(0, y - 90)
                info_x = max(0, x)
            else:  # Place below the food
                info_y = min(frame.shape[0] - 80, y + h + 10)
                info_x = max(0, x)
                
            # Make sure info box fits in frame
            info_x = min(info_x, frame.shape[1] - 200)
            
            # Add info box to frame with semi-transparency
            alpha = 0.7
            roi = frame[info_y:info_y+80, info_x:info_x+200]
            frame[info_y:info_y+80, info_x:info_x+200] = cv2.addWeighted(roi, 1-alpha, info_bg, alpha, 0)
            
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
        print("Press 'q' to quit, 'd' to toggle demo mode")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Cannot read frame from webcam")
                break
                
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect food in the frame (placeholder until model is ready)
            food_items = self.detect_food(frame)
            
            # Draw food information in AR
            if food_items:
                frame = self.draw_food_info(frame, food_items)
                
            # Add AR interface elements
            frame_with_ui = self.add_ar_interface(frame)
            
            # Display the result
            cv2.imshow('Food Calories AR', frame_with_ui)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.demo_mode = not self.demo_mode
                print(f"Demo mode {'enabled' if self.demo_mode else 'disabled'}")
                self.detected_foods = []  # Clear existing detections
                
        # Clean up
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