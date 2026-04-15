
 
"""
3/30/2026
 Three-Body Periodic Solution
Stimkov's Solution, this is in 3d 
Units: AU, Solar Masses, Years  
"""


# Sitnikov Problem Simulation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Time settings
dt       = 0.001
t_finish = 4.44 # scaled to AU/yr units one period
time     = np.arange(0, t_finish, dt)

# Circular orbit parameters for primaries
R     = 1.0
omega = 2 * np.pi

# Initial conditions for third body along z-axis
z      = np.zeros_like(time)
vz     = np.zeros_like(time)
z[0]   = 0.5 # initial z position
vz[0]  = 0.0

# Energy storage
energy    = np.zeros_like(time)
energy[0] = 0.5 * vz[0]**2 - 2 / np.sqrt(R**2 + z[0]**2) # Initial energy of the system, kinetic + potential equation 4 from poster
E0        = energy[0]

# Leapfrog integration
for i in range(len(time) - 1):
    a = -2 * z[i] / (R**2 + z[i]**2)**(3/2) # Acceleration from equation 3 in poster
    vz[i+1] = vz[i] + a * dt # Update velocity
    z[i+1] = z[i] + vz[i+1] * dt # Update position using new velocity 
    energy[i+1] = 0.5 * vz[i+1]**2 - 2 / np.sqrt(R**2 + z[i+1]**2) # Compute energy at new position and velocity

# Relative energy error
rel_energy_err = (energy - E0) / abs(E0)

# Plotting settings
colors  = ['blue', 'orange', 'purple']
colors2 = ['blue', 'orange', 'black']
shapes  = ['o', 'v', 's']
labels  = ['Body 1', 'Body 2', 'Body 3']

ARROW_LEN    = 0.15 # length of velocity arrow
ARROW_OFFSET = 0.1 # gap between arrowhead and final position

# Primaries in circular orbit
x1      = R * np.cos(omega * time)
y1      = R * np.sin(omega * time)
x2      = -x1
y2      = -y1
z_plane = np.zeros_like(time)
z_axis  = np.zeros_like(time)

bodies = [
    (x1,     y1,     z_plane, 'Body 1'),
    (x2,     y2,     z_plane, 'Body 2'),
    (z_axis, z_axis, z,       'Body 3'),
]

# Plotting
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)

# Left plot: Trajectory
for i, (bx, by, bz, lbl) in enumerate(bodies):
    ax1.plot(bx, by, bz, color=colors2[i], lw=2.0, alpha=0.9, label=lbl)

    
# Plot starting points
    ax1.scatter(bx[-1], by[-1], bz[-1],
                color=colors[i], s=120, marker=shapes[i],
                edgecolors='black', linewidths=0.8, zorder=6)

    end_i   = len(bx) - 1
    start_i = max(0, end_i - 40)

    tip       = np.array([bx[end_i],   by[end_i],   bz[end_i]])
    tail      = np.array([bx[start_i], by[start_i], bz[start_i]])
    direction = tip - tail
    norm      = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm

    arrow_start = tip - direction * (ARROW_OFFSET + ARROW_LEN)
    ax1.quiver(*arrow_start, *(direction * ARROW_LEN),
               color=colors[i], linewidth=2.5, arrow_length_ratio=0.5)

ax1.set_xlabel("X (AU)", fontsize=12)
ax1.set_ylabel("Y (AU)", fontsize=12)
ax1.set_zlabel("Z (AU)", fontsize=12)


# Right plot: Energy Drift
ax2.plot(time, rel_energy_err, 'b-', lw=0.8)
ax2.set_xlabel("Time (years)", fontsize=13)
ax2.set_ylabel("Relative Energy Change [(E-E₀)/E₀]", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
