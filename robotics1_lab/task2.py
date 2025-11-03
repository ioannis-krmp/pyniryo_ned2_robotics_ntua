from pyniryo import *
from pyniryo.vision import uncompress_image, undistort_image
import cv2
import numpy as np
import time
import random
import json
import os
import atexit
import signal
import sys

class PositionValidator:
    def __init__(self):
        self.positions_file = "robot_positions.json"
        self.required_positions = [
            'grasp_1', 'grasp_2', 'grasp_3',
            'intermediate_1', 'intermediate_2', 'intermediate_3',
            'release_1', 'release_2', 'release_3'
        ]
    
    def validate_positions_file(self):
        if not os.path.exists(self.positions_file):
            print(f"ERROR: Position file '{self.positions_file}' not found!")
            print("Please run 'position_configurator.py' first to set up positions.")
            return False
        
        try:
            with open(self.positions_file, 'r') as f:
                positions_data = json.load(f)
            
            missing_positions = []
            for pos_name in self.required_positions:
                if pos_name not in positions_data or positions_data[pos_name] is None:
                    missing_positions.append(pos_name)
            
            if missing_positions:
                print(f"ERROR: Missing positions: {missing_positions}")
                print("Please run 'position_configurator.py' to configure missing positions.")
                return False
            
            print("✅ All positions validated successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR: Could not read position file: {e}")
            return False
    
    def load_positions(self):
        try:
            with open(self.positions_file, 'r') as f:
                positions_data = json.load(f)
            
            positions = {}
            for name, pos_list in positions_data.items():
                if pos_list and len(pos_list) >= 6:
                    positions[name] = JointsPosition(
                        pos_list[0], pos_list[1], pos_list[2],
                        pos_list[3], pos_list[4], pos_list[5]
                    )
            
            return positions
        except Exception as e:
            print(f"Error loading positions: {e}")
            return {}

class NiryoNed2Robot:
    def __init__(self):
        self.home_position = JointsPosition(0.0, 0.3, -1.3, 0.0, 0.0, 0.0)
        #self.capture_image_position = JointsPosition(-0.025, 0.093, -0.146, 0.033, -1.853, -0.038)
        self.capture_image_position = JointsPosition(-0.012, 0.297, -0.706, 0.033, -1.277, -0.3)
        self.position_before_grasping = self.capture_image_position
        self.position_before_releasing = JointsPosition(-1.6, -0.185, -0.5, 0.1, -0.91, -0.11)       
        self.robot_ip = "192.168.1.120"
        self.robot = None
        
        self.color_ranges = {
            "red": [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
            "green": [([35, 40, 40], [90, 255, 255])],
            "blue": [([100, 50, 50], [130, 255, 255])],
        }
        
        self.min_area = 300
        self.max_area = 10000
        self.morph_kernel = np.ones((3, 3), np.uint8)
        
        self.mtx = None
        self.dist = None
        self.positions = {}
        
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}. Cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        cv2.destroyAllWindows()
        if self.robot:
            try:
                print("Disconnecting from robot...")
                self.robot.close_connection()
                print("Robot disconnected successfully")
            except:
                print("Error during robot disconnect")
        self.robot = None

    def connect_to_robot(self):
        print(f"Initializing connection with the robot. IP: {self.robot_ip}")
        self.robot = NiryoRobot(self.robot_ip)

    def initialize_robot(self):
        try:
            print("Initializing robot...")
            self.connect_to_robot()
            
            print("Remove collision errors from previous sessions")
            self.robot.clear_collision_detected()

            if self.robot.need_calibration():
                print("Robot calibration procedure")
                self.robot.calibrate_auto()
            else:
                print("Robot is already calibrated")
            
            self.robot.move(self.home_position)
            self.robot.move(self.capture_image_position)
            self.mtx, self.dist = self.robot.get_camera_intrinsics()
            print("Camera calibration loaded successfully")

            self.robot.update_tool()
            print("Tool update")
            
            # Load positions
            validator = PositionValidator()
            self.positions = validator.load_positions()
            
        except Exception as e:
            print(f"Error during robot initialization: {e}")
            self.cleanup()
            raise

    def get_position(self, position_name):
        return self.positions.get(position_name)

    def capture_and_process_image(self):
        img_compressed = self.robot.get_img_compressed()
        img_raw = uncompress_image(img_compressed)
        
        if self.mtx is not None and self.dist is not None:
            img_undistorted = undistort_image(img_raw, self.mtx, self.dist)
        else:
            img_undistorted = img_raw
            
        return img_raw, img_undistorted

    def detect_circles_by_color(self, img, target_colors=None):
        if target_colors is None:
            target_colors = list(self.color_ranges.keys())
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        detected_circles = []
        
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        for color in target_colors:
            if color not in self.color_ranges:
                continue
                
            mask_all = np.zeros(hsv.shape[:2], np.uint8)
            
            for lower, upper in self.color_ranges[color]:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask_all = cv2.bitwise_or(mask_all, mask)
            
            mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, self.morph_kernel, iterations=3)
            mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, self.morph_kernel, iterations=2)
            
            contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                
                if area < self.min_area or area > self.max_area:
                    continue
                
                circle_info = self.analyze_circle_contour(cnt, color, i)
                if circle_info and circle_info["shape"] == "circle":
                    detected_circles.append(circle_info)
        
        return detected_circles

    def analyze_circle_contour(self, contour, color, obj_id):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return None
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity < 0.6:
            return None
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w//2, y + h//2
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        return {
            'id': f"{color}_circle_{obj_id}",
            'color': color,
            'shape': "circle",
            'contour': contour,
            'area': area,
            'center': (cx, cy),
            'bounding_box': (x, y, w, h),
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
        }
    
    def draw_circle_detections(self, img, circles):
        result_img = img.copy()
        
        for i, circle in enumerate(circles):
            cv2.drawContours(result_img, [circle['contour']], -1, (0, 255, 255), 3)
            cv2.circle(result_img, circle['center'], 5, (0, 0, 255), -1)
            
            label = f"{circle['color']} circle"
            cv2.putText(result_img, label, 
                       (circle['center'][0] - 50, circle['center'][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(result_img, str(i + 1),
                       (circle['center'][0] - 10, circle['center'][1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return result_img

    def live_camera_view(self):
        print("Starting live camera view...")
        print("Controls:")
        print("  'q' or ESC: Exit live view")
        print("  'd': Detect circles")
        print("  's': Take screenshot")
        
        detection_mode = False
        
        while True:
            try:
                img_raw, img_processed = self.capture_and_process_image()
                
                if detection_mode:
                    circles = self.detect_circles_by_color(img_processed)
                    display_img = self.draw_circle_detections(img_processed, circles)
                    
                    cv2.putText(display_img, f"Circles detected: {len(circles)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_img, "Detection ON - Press 'd' to toggle", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    display_img = img_processed
                    cv2.putText(display_img, "Live View - Press 'd' for detection", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(display_img, "Press 'q' or ESC to exit", 
                           (10, display_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Niryo Camera Live View', display_img)
                
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('d'):  # Toggle detection
                    detection_mode = not detection_mode
                    mode_text = "ON" if detection_mode else "OFF"
                    print(f"Detection mode: {mode_text}")
                elif key == ord('s'):  # Screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"niryo_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, display_img)
                    print(f"Screenshot saved: {filename}")
                    
            except Exception as e:
                print(f"Error in live view: {e}")
                break
        
        cv2.destroyAllWindows()
        print("Live camera view ended")

    def detect_and_assign_circles(self):
        print("Detecting circles...")
        img_raw, img_processed = self.capture_and_process_image()
        circles = self.detect_circles_by_color(img_processed)
        
        if not circles:
            print("No circles detected!")
            return []
        
        print(f"Found {len(circles)} circles:")
        for i, circle in enumerate(circles):
            print(f"  {i+1}. {circle['color']} circle - Area: {circle['area']:.0f}")
        
        # Show detection result
        result_img = self.draw_circle_detections(img_processed, circles)
        cv2.imshow('Detected Circles', result_img)
        cv2.waitKey(3000)  # Show for 3 seconds
        cv2.destroyAllWindows()
        
        return circles

    def pick_and_place_sequence(self, grasp_num, release_num):
        try:
            grasp_pos = self.get_position(f'grasp_{grasp_num}')
            intermediate_pos = self.get_position(f'intermediate_{grasp_num}')
            release_pos = self.get_position(f'release_{release_num}')
            
            if not all([grasp_pos, intermediate_pos, release_pos]):
                print(f"Missing positions for sequence {grasp_num} -> {release_num}")
                return False

            print(f"Executing pick and place: grasp_{grasp_num} -> release_{release_num}")
            
            self.robot.release_with_tool()
            time.sleep(1)
            
            # Move to intermediate position
            pos_list = intermediate_pos.to_list()
            clean_pos = JointsPosition(pos_list[0], pos_list[1], pos_list[2], pos_list[3], pos_list[4], pos_list[5])
            self.robot.move(clean_pos)
            
            # Move to grasp position
            pos_list = grasp_pos.to_list()
            clean_pos = JointsPosition(pos_list[0], pos_list[1], pos_list[2], pos_list[3], pos_list[4], pos_list[5])
            self.robot.move(clean_pos)
            
            # Close gripper
            self.robot.grasp_with_tool()
            time.sleep(1)
            
            # Move back to intermediate
            pos_list = intermediate_pos.to_list()
            clean_pos = JointsPosition(pos_list[0], pos_list[1], pos_list[2], pos_list[3], pos_list[4], pos_list[5])
            self.robot.move(clean_pos)

            time.sleep(1)
            self.robot.move(self.capture_image_position)
            time.sleep(1)
            self.robot.move(self.position_before_releasing)
            time.sleep(1)

            # Move to release position
            pos_list = release_pos.to_list()
            clean_pos = JointsPosition(pos_list[0], pos_list[1], pos_list[2], pos_list[3], pos_list[4], pos_list[5])
            self.robot.move(clean_pos)
            time.sleep(1)
            # Open gripper
            self.robot.release_with_tool()
            time.sleep(1)
            
            self.robot.move(self.position_before_releasing)
            time.sleep(1)
            # Return to camera position
            self.robot.move(self.capture_image_position)
            time.sleep(1)
            print("Pick and place sequence completed successfully!")
            return True
            
        except Exception as e:
            print(f"Pick and place sequence failed: {e}")
            return False

    def automated_pick_and_place(self):
        print("Starting automated pick and place...")
        
        circles_moved = 0
        max_attempts = 10
        processed_circle_ids = set()
        
        for attempt in range(max_attempts):
            print(f"\n--- Attempt {attempt + 1} ---")
            
            circles = self.detect_and_assign_circles()
            
            if not circles:
                print("No more circles detected - task complete!")
                break
            
            
            circles = [c for c in circles if c['id'] not in processed_circle_ids]

            # Pick a random circle
            selected_circle = random.choice(circles)
            print(f"Selected: {selected_circle['color']} circle")
            
            # Determine position based on circle (simple assignment)
            position_mapping = {"red": 1, "green": 2, "blue": 3}
            grasp_pos = position_mapping.get(selected_circle['color'], 1)
            release_pos = grasp_pos  # Release to same numbered position
            
            if self.pick_and_place_sequence(grasp_pos, release_pos):
                circles_moved += 1
                processed_circle_ids.add(selected_circle['id'])  # Mark as moved
                print(f"Circle {circles_moved} moved successfully!")
                time.sleep(2)  # Brief pause between operations
            else:
                print("Failed to move circle")
                break
        
        print(f"\nAutomated sequence complete! Moved {circles_moved} circles.")

def show_menu():
    print("\n" + "="*50)
    print("NIRYO ROBOT MENU")
    print("="*50)
    print("1. Live Camera View")
    print("2. Detect Circles (Single Shot)")
    print("3. Test Pick and Place (Manual)")
    print("4. Automated Pick and Place")
    print("5. Exit")
    print("="*50)

def main():
    print("="*50)
    print("NIRYO ROBOT MAIN PROGRAM")
    print("="*50)
    
    # Validate positions first
    validator = PositionValidator()
    if not validator.validate_positions_file():
        print("\nPosition validation failed!")
        print("Run: python position_configurator.py")
        return
    
    # Initialize robot
    ned2_robot = NiryoNed2Robot()
    
    try:
        ned2_robot.initialize_robot()
        print("Robot ready for operations!")
        
        # Main menu loop
        while True:
            show_menu()
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                ned2_robot.live_camera_view()
            
            elif choice == "2":
                ned2_robot.detect_and_assign_circles()
            
            elif choice == "3":
                print("Manual pick and place test:")
                try:
                    grasp_num = int(input("Enter grasp position (1-3): "))
                    release_num = int(input("Enter release position (1-3): "))
                    ned2_robot.pick_and_place_sequence(grasp_num, release_num)
                except ValueError:
                    print("Invalid input! Please enter numbers 1-3")
            
            elif choice == "4":
                ned2_robot.automated_pick_and_place()
            
            elif choice == "5":
                print("Exiting program...")
                break
            
            else:
                print("Invalid choice! Please enter 1-5")
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        ned2_robot.cleanup()

if __name__ == "__main__":
    main()