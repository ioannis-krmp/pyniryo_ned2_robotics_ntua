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
        self.capture_image_position = JointsPosition(-0.025, 0.093, -0.146, 0.033, -1.853, -0.038)
        self.robot_ip = "192.168.1.120"
        self.robot = None
        
        self.color_ranges = {
            "red": [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [180, 255, 255])],
            "green": [([35, 40, 40], [90, 255, 255])],
            "blue": [([100, 50, 50], [130, 255, 255])],
        }
        
        self.min_area = 200
        self.max_area = 12000
        self.morph_kernel = np.ones((5, 5), np.uint8)
        
        self.mtx = None
        self.dist = None
        self.positions = {}
        self.nominal_grasp_positions_file = "nominal_grasp_positions.json"
        self.nominal_grasp_positions = None # will be loaded from file or calibrated
        
        self.color_to_release = {
            "red": 3,
            "green": 2,
            "blue": 1
        }

        self.color_mask_config_file = "color_mask_config.json"
        
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def save_nominal_grasp_positions(self, positions):
        with open(self.nominal_grasp_positions_file, "w") as f:
            json.dump(positions, f)
        print(f"Nominal grasp positions saved to {self.nominal_grasp_positions_file}")

    def load_nominal_grasp_positions(self):
        if os.path.exists(self.nominal_grasp_positions_file):
            with open(self.nominal_grasp_positions_file, "r") as f:
                self.nominal_grasp_positions = json.load(f)
            print(f"Nominal grasp positions loaded from {self.nominal_grasp_positions_file}")
        else:
            print("No nominal grasp positions found. Please calibrate first.")
            self.nominal_grasp_positions = None

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
            
            validator = PositionValidator()
            self.positions = validator.load_positions()
            
            self.load_nominal_grasp_positions()
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
    
    def load_mask_config(self):
        with open(self.color_mask_config_file, "r") as f:
            return json.load(f)
        
    def detect_circles_from_config(self, img=None, color_mask_config_file="color_mask_config.json"):
        import json
        if img is None:
            img_raw, img_processed = self.capture_and_process_image()
            img = img_processed
        with open(color_mask_config_file, "r") as f:
            config = json.load(f)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([config['H_low'], config['S_low'], config['V_low']])
        upper = np.array([config['H_high'], config['S_high'], config['V_high']])
        kernel = np.ones((max(1, config['kernel']), max(1, config['kernel'])), np.uint8)
        iterations = config['iterations']
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 200 or area > 12000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.55:
                continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w//2, y + h//2
                center = (cx, cy)
            circles.append({
                'id': f"circle_{i}",
                'center': center,
                'area': area,
                'circularity': circularity
            })

        # --- Assign color to each detected circle ---
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for circle in circles:
            x, y = circle['center']
            patch = hsv_img[max(y-2,0):y+3, max(x-2,0):x+3]
            mean_hsv = np.mean(patch.reshape(-1, 3), axis=0)
            assigned_color = 'unknown'
            for color_name, ranges in self.color_ranges.items():
                for lower, upper in ranges:
                    lower = np.array(lower)
                    upper = np.array(upper)
                    if np.all(mean_hsv >= lower) and np.all(mean_hsv <= upper):
                        assigned_color = color_name
                        break
                if assigned_color != 'unknown':
                    break
            circle['color'] = assigned_color

        return circles

    def draw_circle_detections(self, img, circles):
        result_img = img.copy()
        for i, circle in enumerate(circles):
            center = circle['center']
            radius = int(circle.get('radius', 28))
            cv2.circle(result_img, center, radius, (0, 255, 255), 3)
            cv2.circle(result_img, center, 5, (0, 0, 255), -1)
            label = f"{circle.get('color', 'unknown')} ({circle.get('confidence', 0):.2f})"
            cv2.putText(result_img, label,
                       (center[0] - 50, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_img, str(i + 1),
                       (center[0] - 10, center[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return result_img

    def calibrate_nominal_grasp_positions(self):
        print("\n=== NOMINAL GRASP POSITION CALIBRATION ===")
        print("Place the nominal template with three circles in view.")
        print("Press SPACE to capture image and calibrate, or ESC to abort.")
        while True:
            img_raw, img_processed = self.capture_and_process_image()
            circles = self.detect_circles_from_config(img_processed)
            display_img = self.draw_circle_detections(img_processed, circles)
            cv2.putText(display_img, "Press SPACE to calibrate, ESC to cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Nominal Calibration", display_img)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                print("Calibration cancelled.")
                return
            elif key == 32:  # SPACE
                if len(circles) != 3:
                    print(f"Detected {len(circles)} good circles (confidence >= 0.6). Please ensure exactly 3 circles are visible for calibration.")
                    continue
                sorted_circles = sorted(circles, key=lambda c: c['center'][0])  # left to right
                nominal_positions = [c['center'] for c in sorted_circles]
                self.save_nominal_grasp_positions(nominal_positions)
                self.nominal_grasp_positions = nominal_positions
                cv2.destroyAllWindows()
                print("Calibration complete.")
                return

    def match_circles_to_nominal_positions(self, detected_circles, distance_threshold=50):
        if not self.nominal_grasp_positions:
            print("Nominal grasp positions not set. Please calibrate first.")
            return {}
        assignments = {}
        used_circles = set()
        for idx, nominal_center in enumerate(self.nominal_grasp_positions):
            # Find the closest circle not already assigned
            closest = None
            min_dist = float('inf')
            for c in detected_circles:
                if c['id'] in used_circles:
                    continue
                dist = np.linalg.norm(np.array(c['center']) - np.array(nominal_center))
                if dist < min_dist:
                    min_dist = dist
                    closest = c
            # Only assign if the circle is close enough (avoid assigning random far circles)
            if closest and min_dist < distance_threshold:
                assignments[f'grasp_{idx+1}'] = closest
                used_circles.add(closest['id'])
        return assignments

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
            self.robot.move(intermediate_pos)
            self.robot.move(grasp_pos)
            self.robot.grasp_with_tool()
            time.sleep(1)
            self.robot.move(intermediate_pos)
            self.robot.move(release_pos)
            self.robot.release_with_tool()
            time.sleep(1)
            self.robot.move(self.capture_image_position)
            print("Pick and place sequence completed successfully!")
            return True
        except Exception as e:
            print(f"Pick and place sequence failed: {e}")
            return False

    def automated_pick_and_place(self):
        print("Starting automated pick and place...")
        if not self.nominal_grasp_positions:
            print("Nominal grasp positions not set. Please calibrate first.")
            return
        circles_moved = 0
        max_attempts = 10
        for attempt in range(max_attempts):
            print(f"\n--- Attempt {attempt + 1} ---")
            circles = self.detect_circles_from_config()
            
            if not circles:
                print("No more circles detected - task complete!")
                break
            assignments = self.match_circles_to_nominal_positions(circles)
            processed_circle_ids = set()
            for grasp_key, circle in assignments.items():
                if circle['id'] in processed_circle_ids:
                    continue
                color = circle.get('color','unknown')
                if color == 'unknown':
                    print(f"Cannot assign release for unknown color: {circle}")
                    continue
                grasp_num = int(grasp_key.split('_')[1])
                release_num = self.color_to_release.get(color, 1)
                print(f"Moving {color} circle from {grasp_key} to release_{release_num})")
                success = self.pick_and_place_sequence(grasp_num, release_num)
                if success:
                    circles_moved += 1
                    processed_circle_ids.add(circle['id'])
                    print(f"Circle {circles_moved} moved successfully!")
                    # Remove this circle from further assignments
                    circles = [c for c in circles if c['id'] != circle['id']]
                    time.sleep(2)
                else:
                    print("Failed to move circle")
                    break
        print(f"\nAutomated sequence complete! Moved {circles_moved} circles.")

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
                    #circles = [c for c in self.fused_circle_detection(img_processed) if c.get('confidence',0) >= 0.6]
                    circles = self.detect_circles_from_config(img_processed)
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

def show_menu():
    print("\n" + "="*50)
    print("NIRYO ROBOT MENU")
    print("="*50)
    print("1. Calibrate Nominal Grasp Positions")
    print("2. Live Camera View")
    print("3. Detect Circles (Single Shot)")
    print("4. Automated Pick and Place")
    print("5. Exit")
    print("="*50)

def main():
    print("="*50)
    print("NIRYO ROBOT MAIN PROGRAM")
    print("="*50)
    
    validator = PositionValidator()
    if not validator.validate_positions_file():
        print("\nPosition validation failed!")
        print("Run: python position_configurator.py")
        return
    ned2_robot = NiryoNed2Robot()
    try:
        ned2_robot.initialize_robot()
        print("Robot ready for operations!")
        while True:
            show_menu()
            choice = input("Enter your choice (1-5): ").strip()
            if choice == "1":
                ned2_robot.calibrate_nominal_grasp_positions()
            elif choice == "2":
                ned2_robot.live_camera_view()
            elif choice == "3":
                ned2_robot.detect_circles_from_config()
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