import cv2
import numpy as np
import time
import json
from pyniryo import *
from pyniryo.vision import uncompress_image, undistort_image

def nothing(x):
    pass

def capture_and_process_image():
    robot = NiryoRobot("192.168.1.120")
    img_compressed = robot.get_img_compressed()
    img_raw = uncompress_image(img_compressed)
    mtx, dist = robot.get_camera_intrinsics()
    if mtx is not None and dist is not None:
        img_undistorted = undistort_image(img_raw, mtx, dist)
    else:
        img_undistorted = img_raw
    return img_raw, img_undistorted

def create_mask_trackbars(window_name):
    cv2.createTrackbar('H_low', window_name, 0, 179, nothing)
    cv2.createTrackbar('S_low', window_name, 0, 255, nothing)
    cv2.createTrackbar('V_low', window_name, 0, 255, nothing)
    cv2.createTrackbar('H_high', window_name, 179, 179, nothing)
    cv2.createTrackbar('S_high', window_name, 255, 255, nothing)
    cv2.createTrackbar('V_high', window_name, 255, 255, nothing)
    cv2.createTrackbar('Kernel', window_name, 3, 20, nothing)
    cv2.createTrackbar('Iterations', window_name, 1, 10, nothing)

def get_mask_params(window_name):
    h_low = cv2.getTrackbarPos('H_low', window_name)
    s_low = cv2.getTrackbarPos('S_low', window_name)
    v_low = cv2.getTrackbarPos('V_low', window_name)
    h_high = cv2.getTrackbarPos('H_high', window_name)
    s_high = cv2.getTrackbarPos('S_high', window_name)
    v_high = cv2.getTrackbarPos('V_high', window_name)
    kernel = cv2.getTrackbarPos('Kernel', window_name)
    iterations = cv2.getTrackbarPos('Iterations', window_name)
    return h_low, s_low, v_low, h_high, s_high, v_high, kernel, iterations

def live_camera_view():
    print("Starting live camera view with color mask (trackbars)...")
    print("Controls:")
    print("  'q' or ESC: Exit live view")
    print("  's': Take screenshot")

    window_name = 'Niryo Camera Live View'
    cv2.namedWindow(window_name)
    create_mask_trackbars(window_name)

    while True:
        try:
            img_raw, img_processed = capture_and_process_image()

            # Get mask parameters from trackbars
            h_low, s_low, v_low, h_high, s_high, v_high, kernel, iterations = get_mask_params(window_name)
            hsv = cv2.cvtColor(img_processed, cv2.COLOR_BGR2HSV)
            lower = np.array([h_low, s_low, v_low])
            upper = np.array([h_high, s_high, v_high])
            mask = cv2.inRange(hsv, lower, upper)
            k = np.ones((max(1, kernel), max(1, kernel)), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=iterations)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Show original and mask side by side
            display_img = np.hstack((img_processed, mask_bgr))
            cv2.putText(display_img, "Live View + Color Mask (trackbars)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_img, "Press 'q' or ESC to exit, 's' to screenshot", 
                        (10, display_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"niryo_mask_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, display_img)
                print(f"Screenshot saved: {filename}")
            elif key == ord('p'):
                print(f"H_low: {h_low}, S_low: {s_low}, V_low: {v_low}")
                print(f"H_high: {h_high}, S_high: {s_high}, V_high: {v_high}")
                print(f"Kernel: {kernel}, Iterations: {iterations}")
            elif key == ord('h'):
                settings = {
                    "H_low": h_low, "S_low": s_low, "V_low": v_low,
                    "H_high": h_high, "S_high": s_high, "V_high": v_high,
                    "kernel": kernel, "iterations": iterations
                }
                with open("color_mask_config.json", "w") as f:
                    json.dump(settings, f, indent=2)
                print("Settings saved to color_mask_config.json")

        except Exception as e:
            print(f"Error in live view: {e}")
            break

    cv2.destroyAllWindows()
    print("Live camera view ended")

if __name__ == "__main__":
    live_camera_view()