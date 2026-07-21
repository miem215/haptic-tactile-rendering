import numpy as np
import mujoco

class TactileHapticController:
    def __init__(self, model, data, site_name="tip_site"):
        self.model = model
        self.data = data
        self.site_id = model.site(site_name).id
        
        # Kinesthetic Spring-Damper parameters
        self.K_spring = np.diag([120.0, 120.0])  
        self.B_damper = np.diag([6.0, 6.0])     
        
        # Physical boundary matching XML left face (0.30 - 0.05 = 0.25m)
        self.wall_x = 0.3                   
        self.K_wall = 350.0                    
        self.B_wall = 12.0                     
        
        self.target_pos = np.array([0.285, 0.0]) # Pushes slightly INTO wall (X=0.27)
        
        # Cutaneous / Tactile Texture Parameters
        self.active_material = "decay" # Options: 'enamel', 'decay', 'tissue'
        
        self.material_profiles = {
            "enamel": {"stiffness": 500.0, "mu": 0.05, "spatial_freq": 10.0, "vib_amp": 0.2},
            "decay":  {"stiffness": 800.0, "mu": 0.45, "spatial_freq": 80.0, "vib_amp": 2.5},
            "tissue": {"stiffness": 80.0,  "mu": 0.80, "spatial_freq": 25.0, "vib_amp": 0.8}
        }

    def set_material(self, mat_name):
        if mat_name in self.material_profiles:
            self.active_material = mat_name

    def get_end_effector_state(self):
        pos_xy = self.data.site_xpos[self.site_id][:2]
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)
        J_xy = jacp[:2, :]
        vel_xy = J_xy @ self.data.qvel
        return pos_xy, vel_xy, J_xy

    def compute_tactile_forces(self, pos, vel):
    # 1. Base Virtual Spring pulling toward target
        force_xy = -self.K_spring @ (pos - self.target_pos) - self.B_damper @ vel
        
        # Account for Stylus Tip Radius (1.5 cm = 0.015 m)
        tip_radius = 0.015
        surface_contact_x = self.wall_x - tip_radius  # 0.25 - 0.015 = 0.235m
        
        # 2. Virtual Wall Contact
        if pos[0] >= surface_contact_x:
            penetration = pos[0] - surface_contact_x
            mat = self.material_profiles[self.active_material]
            
            # Calculate Normal Force Magnitude (Stiffness + Damping)
            f_normal_mag = mat["stiffness"] * penetration + self.B_wall * max(0.0, vel[0])
            
            # Repelling Force: MUST push back toward -X (left)
            force_xy[0] = -abs(f_normal_mag)  
            
            # Cutaneous Component A: Friction opposing Y motion
            f_friction = -mat["mu"] * abs(f_normal_mag) * np.tanh(10.0 * vel[1])
            
            # Cutaneous Component B: High-Frequency Spatial Micro-Texture
            f_tactile_vib = mat["vib_amp"] * np.sin(2.0 * np.pi * mat["spatial_freq"] * pos[1])
            
            # Add friction & texture to Y-axis
            force_xy[1] += (f_friction + f_tactile_vib)

        return force_xy

    def update_torques(self):
        pos, vel, J = self.get_end_effector_state()
        force_xy = self.compute_tactile_forces(pos, vel)
        
        tau = J.T @ force_xy
        ctrl_range = self.model.actuator_ctrlrange
        data_ctrl = np.clip(tau, ctrl_range[:, 0], ctrl_range[:, 1])
        
        return data_ctrl, force_xy