 

"""
    James Ward
    3/28/26
    This Program examines the three-body problem initial conditions are set up for the collinear results.
    Masses are scaled in solar masses and distances are scaled in Astronomical Units.
"""


# Import libraries
import matplotlib.pyplot as plt # Need this for plotting
import matplotlib.style 
import numpy as np # Need this for math


# Gravitational constant in terms of AU, Derived from Keplers 3rd Law, from Equation 1
G = 4 * np.pi**2

# Time step 
dt = 1/1000

# Total simulation time in years 
t_finish = 2.53   # Finish Time
time = 0.0 # initial Time


# Mass of planets stored in list
# This is in terms of solar mass so the sun is 1.0
masses = [1, 1, 1]           

# For collinear masses need to be equal and start across from eachother on a line
d = 2.0          # separation in AU
m = 1.0          # equal masses solar masses


# angular velocity for bodies need to be similar for outer bodies in circular orbits
omega = np.sqrt(5 * G * m / (4 * d**3))



# Positions: -d, 0, +d along x-axis
positions = [
    np.array([-d, 0.0]),    # 
    np.array([0.0, 0.0]),   # Central
    np.array([d,  0.0])     # 
]

# Outer bodies orbit
velocities = [
    np.array([0.0, -omega * d]),   # Body 1
    np.array([0.0,  0.0]),         # Body 2 (zero force, zero velocity)
    np.array([0.0,  omega * d])    # Body 3
]

# number of bodies equals how many values are in masses
n_bodies = len(masses)


# Store trajectories in array
trajectories = [[] for _ in range(n_bodies)]


# append trajectory from positions
for i in range(n_bodies):
    trajectories[i].append(positions[i].copy())

# Energy Conservation Check, use this to see if energy is being conserved

energy_time = []      # time values for energy plot
total_energy = []     # total energy at each time
total_energyrel = []


# Total Energy equals kinetic Energy + Potential Energy 
# Equation 4 from poster is both KE and PE this is the application of the Total Energy Equation
def total_energylist(pos, vel):
    
    KE = 0.0 # 
    for i in range(n_bodies):
        KE += 0.5 * masses[i] * np.dot(vel[i], vel[i]) # equation 4 first term from Poster.
    PE = 0.0
    for i in range(n_bodies):
        for j in range(i+1, n_bodies):
            r_vec = pos[j] - pos[i] # position vector
            r = np.linalg.norm(r_vec) # magnitude of position vector
            PE -= G * masses[i] * masses[j] / r # Potential Energy equation term 2 in equation 4.
    return KE + PE


E0 = total_energylist(positions, velocities)
energy_time.append(time) # Array to keep track of time
total_energy.append(E0)  # Stores total Energy throughout time



# Function to compute accelerations from current positions
def accelerations(pos):
    acc = [np.zeros(2) for _ in range(n_bodies)] # start with no acceleration, sets initial accelerations to zero
    for i in range(n_bodies):   # 
        for j in range(n_bodies):
            if i == j: # can't exert force on itself, make sure its not trying to 
                continue
            r_vec = pos[j] - pos[i] # computes vector pointing towards j, 
            r = np.linalg.norm(r_vec) # distance between i and j, computes magnitude
            acc[i] = acc[i] + G * masses[j] * r_vec / r**3 # Equation 3 from Poster
    return acc

# Initial accelerations (needed for leapfrog start)
acc = accelerations(positions) # compute initial acceleration using acceleration function


E0 = total_energylist(positions, velocities)

# Simulation loop Leapfrog Method, Equations 5 and 6 from Poster
while time <= t_finish:
    #  Half velocity using current acceleration
    for i in range(n_bodies):
        velocities[i] = velocities[i] + 0.5 * dt * acc[i]#equation 6 from poster

    # Full step position using the half step
    for i in range(n_bodies):
        positions[i] = positions[i] + dt * velocities[i] 

    # Compute new acceleration using new positions
    new_acc = accelerations(positions)

    # Second half step using new acceleration
    for i in range(n_bodies):
        velocities[i] = velocities[i] + 0.5 * dt * new_acc[i] # 2.7

    # Use trajectories array to store the positions
    for i in range(n_bodies):
        trajectories[i].append(positions[i].copy())


    time += dt

    # --- Append energy and time AFTER update ---
    E = total_energylist(positions, velocities)
    energy_time.append(time)                     # same length as total_energyrel
    total_energyrel.append((E - E0)/E0)
    
    acc = new_acc


# Convert trajectories to arrays for plotting
traj = [np.array(t) for t in trajectories]




# Setup for Figure 1
plt.figure(figsize=(12,10))

#label Bodies
labels = ['Body 1', 'Body 2', 'Body 3']


# Figure size and energy and trajectory on one figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Stores end of plot
x_ends, y_ends = [], []
# Color of each body
colors = ['blue', 'orange', 'purple']


# Plots
colors = ['blue', 'orange', 'purple']   # Colors for the three bodies in the trajectory plot    
labels = ['Body 1', 'Body 2', 'Body 3'] # Labels for the three bodies in the trajectory plot
colors2 = ['blue', 'orange', 'black']  # Colors for the trajectory lines, with the third body in black for better visibility
shapes = ['o', 'd', 's']               # Marker shapes 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')    # Create a figure with two subplots side by side, one for the trajectory and one for the energy drift, with a white background



 
ARROW_LEN    = 0.15   # bigger arrow
ARROW_OFFSET = 0.1   # gap between arrowhead and final position


# plot trajectories and energy drift
for i in range(n_bodies):
    x_ends.append(traj[i][-1, 0])
    y_ends.append(traj[i][-1, 1])
    ax1.plot(traj[i][:, 0], traj[i][:, 1],
            color=colors2[i], lw=2.0, alpha=0.9, label=labels[i])

    ax1.scatter(traj[i][0, 0], traj[i][0, 1],
               color=colors[i], s=120, zorder=5,
               edgecolors='black', linewidths=0.8, marker=shapes[i],
               label=f'{labels[i]} Start')

    end_i   = len(traj[i]) - 1
    start_i = max(0, end_i - 40)

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
x_ends = np.array(x_ends)
y_ends = np.array(y_ends)
if np.max(x_ends) - np.min(x_ends) < 1e-6:
    ax1.axvline(x=x_ends[0], color='red', linestyle='--', linewidth=2)
else:
    m, b = np.polyfit(x_ends, y_ends, 1)
    x_line = np.linspace(min(x_ends) - 1, max(x_ends) + 1, 100)
    ax1.plot(x_line, m * x_line + b, 'r--', linewidth=2)
# Set limits, labels, grid
ax1.set_xlabel('x (AU)')
ax1.set_ylabel('y (AU)')
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.5)

# Right plot, Energy Drift
ax2.plot(energy_time[1:], total_energyrel,'b-')
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Relative Energy Change [(E-E0)/E0]')
ax2.grid(True)
plt.tight_layout()
plt.show()
