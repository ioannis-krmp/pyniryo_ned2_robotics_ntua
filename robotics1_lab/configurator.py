from pyniryo import *
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import time
import atexit
import signal
import sys

class PositionManager:
    def __init__(self, robot):
        self.robot = robot
        self.positions_file = "robot_positions.json"
        self.positions = {
            'grasp_1': None, 'grasp_2': None, 'grasp_3': None,
            'intermediate_1': None, 'intermediate_2': None, 'intermediate_3': None,
            'release_1': None, 'release_2': None, 'release_3': None
        }
        self.load_positions()
        
    def save_position(self, position_name):
        try:
            current_joints = self.robot.get_joints()
            self.positions[position_name] = current_joints
            self.save_to_file()
            print(f"Saved {position_name}: {current_joints}")
            return True
        except Exception as e:
            print(f"Error saving position {position_name}: {e}")
            return False
    
    def get_position(self, position_name):
        return self.positions.get(position_name)
    
    def save_to_file(self):
        try:
            positions_data = {}
            for name, pos in self.positions.items():
                if pos:
                    positions_data[name] = pos.to_list()
                else:
                    positions_data[name] = None
            
            with open(self.positions_file, 'w') as f:
                json.dump(positions_data, f, indent=2)
            print(f"Positions saved to {self.positions_file}")
        except Exception as e:
            print(f"Error saving to file: {e}")
    
    def load_positions(self):
        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file, 'r') as f:
                    positions_data = json.load(f)
                
                for name, pos_list in positions_data.items():
                    if pos_list and len(pos_list) >= 6:
                        self.positions[name] = JointsPosition(
                            pos_list[0], pos_list[1], pos_list[2],
                            pos_list[3], pos_list[4], pos_list[5]
                        )
                print("Positions loaded from file")
            except Exception as e:
                print(f"Error loading positions: {e}")
    
    def has_all_positions(self):
        return all(pos is not None for pos in self.positions.values())

class PositionGUI:
    def __init__(self, position_manager):
        self.position_manager = position_manager
        self.window = tk.Tk()
        self.window.title("Niryo Position Configurator")
        self.window.geometry("800x800")
        self.status_labels = {}
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.window, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(main_frame, text="Robot Position Configurator", font=("Arial", 18, "bold")).grid(row=0, column=0, columnspan=3, pady=15)
        
        instructions = ttk.Label(main_frame, text="Move robot to desired position using free motion, then click Save", 
                               font=("Arial", 11), foreground="blue")
        instructions.grid(row=1, column=0, columnspan=3, pady=10)
        
        self.create_position_section(main_frame, "Grasp Positions", ["grasp_1", "grasp_2", "grasp_3"], 2)
        self.create_position_section(main_frame, "Intermediate Positions", ["intermediate_1", "intermediate_2", "intermediate_3"], 3)
        self.create_position_section(main_frame, "Release Positions", ["release_1", "release_2", "release_3"], 4)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=30)
        
        button_row1 = ttk.Frame(button_frame)
        button_row1.pack(pady=5)
        
        button_row2 = ttk.Frame(button_frame)
        button_row2.pack(pady=5)
        
        ttk.Button(button_row1, text="Auto Generate Intermediate", command=self.auto_generate_intermediate, width=20).pack(side=tk.LEFT, padx=8)
        ttk.Button(button_row1, text="Test All Positions", command=self.test_positions, width=20).pack(side=tk.LEFT, padx=8)
        
        ttk.Button(button_row2, text="Show Current Position", command=self.show_current_position, width=20).pack(side=tk.LEFT, padx=8)
        ttk.Button(button_row2, text="Save & Exit", command=self.save_and_exit, width=20).pack(side=tk.LEFT, padx=8)
        
        self.update_status()
        
    def create_position_section(self, parent, title, positions, row):
        frame = ttk.LabelFrame(parent, text=title, padding="15")
        frame.grid(row=row, column=0, columnspan=3, pady=15, sticky=(tk.W, tk.E))
        
        for i, pos_name in enumerate(positions):
            ttk.Label(frame, text=f"Position {i+1}:", font=("Arial", 10)).grid(row=i, column=0, sticky=tk.W, padx=8, pady=5)
            
            status_label = ttk.Label(frame, text="✗ Not Set", foreground="red", font=("Arial", 10))
            status_label.grid(row=i, column=1, sticky=tk.W, padx=15, pady=5)
            self.status_labels[pos_name] = status_label
            
            save_btn = ttk.Button(frame, text=f"Save", width=12,
                                command=lambda name=pos_name: self.save_position(name))
            save_btn.grid(row=i, column=2, padx=8, pady=5)
    
    def show_current_position(self):
        try:
            current_joints = self.position_manager.robot.get_joints()
            joint_list = current_joints.to_list()
            info = f"Current Joint Position:\n"
            for i, value in enumerate(joint_list):
                info += f"J{i+1}: {value:.4f}\n"
            
            messagebox.showinfo("Current Position", info)
        except Exception as e:
            messagebox.showerror("Error", f"Error getting current position: {e}")
    
    def save_position(self, position_name):
        if self.position_manager.save_position(position_name):
            self.status_labels[position_name].config(text="✓ Saved", foreground="green")
            messagebox.showinfo("Success", f"Position {position_name} saved successfully!")
        else:
            messagebox.showerror("Error", f"Failed to save position {position_name}")
    
    def auto_generate_intermediate(self):
        generated = 0
        for i in range(1, 4):
            grasp_pos = self.position_manager.positions.get(f'grasp_{i}')
            if grasp_pos:
                grasp_list = grasp_pos.to_list()
                grasp_list[2] += 0.1
                self.position_manager.positions[f'intermediate_{i}'] = JointsPosition(
                    grasp_list[0], grasp_list[1], grasp_list[2],
                    grasp_list[3], grasp_list[4], grasp_list[5]
                )
                generated += 1
        
        if generated > 0:
            self.position_manager.save_to_file()
            self.update_status()
            messagebox.showinfo("Success", f"Generated {generated} intermediate positions")
        else:
            messagebox.showwarning("Warning", "No grasp positions found to generate intermediate positions")
    
    def test_positions(self):
        try:
            for pos_name, position in self.position_manager.positions.items():
                if position:
                    print(f"Testing position {pos_name}")
                    
                    pos_list = position.to_list()
                    clean_pos = JointsPosition(pos_list[0], pos_list[1], pos_list[2], pos_list[3], pos_list[4], pos_list[5])
                    
                    self.position_manager.robot.move(clean_pos)
                    time.sleep(2)
                    
            messagebox.showinfo("Success", "All positions tested successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Position test failed: {e}")
    
    def save_and_exit(self):
        self.position_manager.save_to_file()
        self.window.destroy()
    
    def update_status(self):
        for pos_name, status_label in self.status_labels.items():
            if self.position_manager.positions[pos_name]:
                status_label.config(text="✓ Set", foreground="green")
            else:
                status_label.config(text="✗ Not Set", foreground="red")
    
    def show(self):
        self.window.mainloop()

class RobotConfigurator:
    def __init__(self):
        self.home_position = JointsPosition(0.0, 0.3, -1.3, 0.0, 0.0, 0.0)
        self.capture_image_position = JointsPosition(-0.025, 0.093, -0.146, 0.033, -1.853, -0.038)
        self.robot_ip = "192.168.1.120"
        self.robot = None
        self.position_manager = None
        
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}. Cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
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
            print("Robot initialized successfully")
            
            self.position_manager = PositionManager(self.robot)
        except Exception as e:
            print(f"Error during robot initialization: {e}")
            self.cleanup()
            raise

def main():
    print("="*50)
    print("NIRYO ROBOT POSITION CONFIGURATOR")
    print("="*50)
    
    configurator = RobotConfigurator()
    
    try:
        configurator.initialize_robot()
        
        print("Opening Position Configuration GUI...")
        gui = PositionGUI(configurator.position_manager)
        gui.show()
        
        print("Configuration complete!")
        
    except KeyboardInterrupt:
        print("\nConfiguration interrupted by user")
    except Exception as e:
        print(f"Configuration error: {e}")
    finally:
        configurator.cleanup()

if __name__ == "__main__":
    main()