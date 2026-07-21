import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

def plot_tactile_signals(log_data, wall_x):
    time = np.array(log_data['time'])
    pos_x = np.array(log_data['pos_x'])
    pos_y = np.array(log_data['pos_y'])
    vel_y = np.array(log_data['vel_y'])
    force_x = np.array(log_data['force_x'])
    force_y = np.array(log_data['force_y'])

    fig, axs = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle('Cutaneous & Kinesthetic Signal Analysis Dashboard', fontsize=14, fontweight='bold')

    t_max = time[-1] if len(time) > 0 else 10.0

    # Panel 1: Position Tracking (Time Domain)
    axs[0].plot(time, pos_x, color='#1f77b4', linewidth=1.5, label='Pos X (Depth)')
    axs[0].plot(time, pos_y, color='#ff7f0e', linewidth=1.5, label='Pos Y (Sliding)')
    axs[0].axhline(y=wall_x, color='red', linestyle='--', linewidth=1.2, label=f'Surface Wall ({wall_x}m)')
    axs[0].set_xlim([0, t_max])
    axs[0].set_ylabel('Position (m)')
    axs[0].set_title('End-Effector Trajectory')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='upper right')

    # Panel 2: Force Modulation (Time Domain)
    axs[1].plot(time, force_x, color='#1f77b4', alpha=0.7, label='Fx: Normal Force (Stiffness)')
    axs[1].plot(time, force_y, color='#2ca02c', linewidth=1.2, label='Fy: Tangential Force (Texture + Friction)')
    axs[1].set_xlim([0, t_max])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Force (N)')
    axs[1].set_title('Force Signals (Cutaneous Friction Modulation)')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='upper right')

    # --- PANEL 3: CUMULATIVE SPATIAL DOMAIN FFT ---
    # 1. Filter data to active sliding contact regions (vel_y > 0.005 m/s, X near wall)
    surface_contact_x = wall_x - 0.015  # Account for 1.5 cm tip radius
    contact_mask = (np.abs(vel_y) > 0.005) & (pos_x >= surface_contact_x - 0.002)

    pos_y_c = pos_y[contact_mask]
    force_y_c = force_y[contact_mask]

    if len(pos_y_c) > 100:
        # 2. Compute Cumulative Spatial Path: s(t) = sum(|delta_y|)
        # This unwraps back-and-forth oscillations into a continuous total distance traveled
        dy_steps = np.abs(np.diff(pos_y_c, prepend=pos_y_c[0]))
        s_path = np.cumsum(dy_steps)  # Monotonically increasing distance (0 to multi-meters)

        # 3. Remove non-unique spatial step duplicates for clean interpolation
        s_path, unique_indices = np.unique(s_path, return_index=True)
        force_y_c = force_y_c[unique_indices]

        # 4. Resample onto a uniform spatial grid (dy = 0.5 mm = 0.0005 m)
        dy = 0.0005  
        s_uniform = np.arange(s_path[0], s_path[-1], dy)

        interp_func = interpolate.interp1d(s_path, force_y_c, kind='linear')
        f_spatial = interp_func(s_uniform)

        # 5. Zero-padded FFT for fine frequency bin density
        N_pad = 8192  # High-density zero-padding
        f_spatial_detrended = f_spatial - np.mean(f_spatial)
        spatial_fft = np.abs(np.fft.rfft(f_spatial_detrended, n=N_pad))
        spatial_freqs = np.fft.rfftfreq(N_pad, d=dy)  # Output in cycles/meter

        # Plot Spatial Spectrum
        axs[2].plot(spatial_freqs, spatial_fft, color='#d62728', linewidth=1.5, label='Spatial Power Spectrum')
        axs[2].axvline(x=80.0, color='black', linestyle=':', linewidth=1.5, label='True Texture Frequency (80 cycles/m)')
        axs[2].set_xlim([0, 200])  # Focus on 0 - 200 cycles/meter
        axs[2].set_xlabel('Spatial Frequency (cycles / meter)')
        axs[2].set_ylabel('Magnitude')
        axs[2].set_title('Spatial Domain Spectrum (Cumulative Distance Path)')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc='upper right')
    else:
        axs[2].text(0.5, 0.5, 'Insufficient Contact Data for Spatial Resampling', 
                     horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)

    plt.tight_layout()
    plt.show()