import time
import pandas as pd
import numpy as np
import mujoco
import mujoco.viewer

from haptic_controller import TactileHapticController
from plot_haptics import plot_tactile_signals

def main():
    model = mujoco.MjModel.from_xml_path("2dof_haptic_device.xml")
    data = mujoco.MjData(model)
    controller = TactileHapticController(model, data)

    # Joint 1 flexed at 45 deg, Joint 2 folded back at -90 deg
    data.qpos[0] = np.radians(45.0)   
    data.qpos[1] = np.radians(-90.0)  
    
    # Forward kinematics update to establish clean starting state
    mujoco.mj_forward(model, data)

    # Logging dictionary
    logs = {
        'time': [], 'pos_x': [], 'pos_y': [], 'vel_x': [], 'vel_y': [],
        'force_x': [], 'force_y': [], 'material': []
    }

    print("--- Running Multimodal Tactile Surface Rendering ---")
    print("Sweeping probe across 'decay' material profile...")

    controller.set_material("decay") # Set material for cutaneous test

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            t = time.time() - start_time
            
            # Drive stylus up and down Y-axis across surface
    
            controller.target_pos[1] = 0.03 * np.sin(2.0 * t)
            
            mujoco.mj_forward(model, data)
            torques, force_xy = controller.update_torques()
            
            data.ctrl[0] = torques[0]
            data.ctrl[1] = torques[1]
            
            # Logging Telemetry
            pos, vel, _ = controller.get_end_effector_state()
            logs['time'].append(t)
            logs['pos_x'].append(pos[0])
            logs['pos_y'].append(pos[1])
            logs['vel_x'].append(vel[0])
            logs['vel_y'].append(vel[1])
            logs['force_x'].append(force_xy[0])
            logs['force_y'].append(force_xy[1])
            logs['material'].append(controller.active_material)

            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Maintain real-time loop speed
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Save to CSV for Machine Learning Pipeline
    df = pd.DataFrame(logs)
    df.to_csv("tactile_surface_dataset.csv", index=False)
    print("Dataset saved to 'tactile_surface_dataset.csv'")

    # Plot Signal Dashboard
    plot_tactile_signals(logs, controller.wall_x)

if __name__ == "__main__":
    main()