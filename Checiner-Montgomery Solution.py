

""" 
3/30/2026
 Three-Body Periodic Solution
Chenciner & Montgomery Periodic Solution
Units: AU, Solar Masses, Years  
"""


# Import Libraries
import numpy as np
import matplotlib.pyplot as plt


# Gravitational constant in AU, Solar Mass, Year units
G = 4 * np.pi**2

# Time step in years
dt = 1 / 1000

# Figure-8 period scaled to AU/yr units
T_period = 6.3259 / (2 * np.pi)
t_finish = T_period
time = 0.0

# Equal masses required for the figure-8 solution 
masses = [1.0, 1.0, 1.0] # Solar Mass




# Initial positions in AU, These are Specific known initial conditions required for the Chenciner & Montgomery Solution
positions = [
    np.array([-0.97000436,  0.24308753, 0.0]),
    np.array([ 0.0,         0.0,        0.0]),
    np.array([ 0.97000436, -0.24308753, 0.0]),
]


# Initial velocities in AU/yr
_vx = 0.93240737 * 2 * np.pi
_vy = 0.86473146 * 2 * np.pi
velocities = [
    np.array([ _vx / 2,  _vy / 2, 0.0]),
    np.array([-_vx,     -_vy,     0.0]),
    np.array([ _vx / 2,  _vy / 2, 0.0]),
]


# number of bodies
n_bodies = len(masses)



# Store trajectories
trajectories = [[] for _ in range(n_bodies)]
for i in range(n_bodies):
    trajectories[i].append(positions[i].copy())



# Energy conservation Arrays
energy_time  = []
total_energy = []


# energy conservation Function
def total_energylist(pos, vel):
    """Compute total mechanical energy (kinetic + potential)"""
    KE = 0.0
    for i in range(n_bodies):
        KE += 0.5 * masses[i] * np.dot(vel[i], vel[i]) # Kinetic Energy
    PE = 0.0
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            r_vec = pos[j] - pos[i] # Vector from body i to j
            r = np.linalg.norm(r_vec)  # Distance between bodies i and j
            PE -= G * masses[i] * masses[j] / r # Potential Energy
    return KE + PE


# Acceleration Function
def accelerations(pos):
    acc = [np.zeros(3) for _ in range(n_bodies)]
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i == j:
                continue
            r_vec = pos[j] - pos[i] # Vector from body i to j
            r = np.linalg.norm(r_vec)   # Distance between bodies i and j
            acc[i] = acc[i] + G * masses[j] * r_vec / r**3 # Newton's law of gravitation for acceleration
    return acc


# Initial energy and accelerations
E0 = total_energylist(positions, velocities)    # Initial total energy
energy_time.append(time)        # Store initial time for energy plot
total_energy.append(E0)     # Store initial energy for energy plot
acc = accelerations(positions)      # Initial accelerations based on initial positions


# Simulation loop Leapfrog method applying equations 5 and 6 from poster
while time <= t_finish:
    for i in range(n_bodies):
        velocities[i] = velocities[i] + 0.5 * dt * acc[i]
    for i in range(n_bodies):
        positions[i] = positions[i] + dt * velocities[i]

    new_acc = accelerations(positions)

    for i in range(n_bodies):
        velocities[i] = velocities[i] + 0.5 * dt * new_acc[i]

    for i in range(n_bodies):
        trajectories[i].append(positions[i].copy())

    E = total_energylist(positions, velocities)
    energy_time.append(time + dt)
    total_energy.append(E)

    time = time + dt
    acc = new_acc

# Convert trajectories to arrays for plotting
traj = [np.array(t) for t in trajectories]      # Convert list of trajectories to numpy arrays for easier plotting
total_energy = np.array(total_energy)       # Convert total energy list to numpy array for easier calculations
energy_time  = np.array(energy_time)        # Convert energy time list to numpy array for easier calculations
rel_drift    = np.abs((total_energy - E0) / E0)         # Relative energy drift for plotting, showing how much the total energy deviates from the initial energy over time



# Plots
colors = ['blue', 'orange', 'purple']   # Colors for the three bodies in the trajectory plot    
labels = ['Body 1', 'Body 2', 'Body 3'] # Labels for the three bodies in the trajectory plot
colors2 = ['blue', 'orange', 'black']  # Colors for the trajectory lines, with the third body in black for better visibility
shapes = ['o', 'd', 's']               # Marker shapes 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')    # Create a figure with two subplots side by side, one for the trajectory and one for the energy drift, with a white background



ARROW_LEN    = 0.15   # bigger arrow
ARROW_OFFSET = 0.1   # gap between arrowhead and final position

# Plot trajectories
for i in range(n_bodies):
    ax1.plot(traj[i][:, 0], traj[i][:, 1],
            color=colors2[i], lw=2.0, alpha=0.9, label=labels[i])
# Plot starting points
    ax1.scatter(traj[i][0, 0], traj[i][0, 1],
               color=colors[i], s=120, zorder=5,
               edgecolors='black', linewidths=0.8, marker=shapes[i],
               label=f'{labels[i]} Start')

    end_i   = len(traj[i]) - 1
    start_i = max(0, end_i - 40)
# Arrow at end of trajectory to show direction of motion
    tip       = traj[i][end_i,   :2]
    tail      = traj[i][start_i, :2]
    direction = tip - tail
    norm      = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    arrow_tip  = tip - direction * ARROW_OFFSET
    arrow_tail = arrow_tip - direction * ARROW_LEN
# arrow at end of trajectory to show direction of motion
    ax1.annotate("",
        xy=arrow_tip,
        xytext=arrow_tail,
        arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=2.5, mutation_scale=20),
        zorder=3,
    )

# Set limits, labels, grid, and legend for the trajectory plot
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-0.75, 0.75)
ax1.set_xlabel("x  [AU]", fontsize=12)
ax1.set_ylabel("y  [AU]", fontsize=12)

# Set ticks, grid, and legend for the trajectory plot
ax1.tick_params(labelsize=10)
ax1.grid(True, color='#dddddd', linewidth=0.5, linestyle='--')

# Right panel: energy drift
ax2.set_facecolor('white')
ax2.plot(energy_time, rel_drift, color='steelblue', lw=1.0)

# Set limits, labels, grid, and legend for the energy drift plot
ax2.set_xlabel("Time  [yr]", fontsize=12)
ax2.set_ylabel("Relative Energy Change  [(E-E0)/E0]", fontsize =12)
ax2.tick_params(labelsize=10)
ax2.set_yscale('log')
ax2.grid(True, color='#dddddd', linewidth=0.5, linestyle='--')


# Adjust layout and save the figure
plt.tight_layout()
plt.show()
