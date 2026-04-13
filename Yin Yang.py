





"""
James Ward
3/31/26
Yin-Yang  Orbit Simulation (Three-Body Problem)
\I used G = 1 to simplify the equations and make the code cleaner, but the solution is still 
valid in physical units as long as the initial conditions are scaled appropriately. The key is that the initial
conditions (positions and velocities) must be chosen to satisfy the equations of motion under the chosen units. 
The specific initial conditions for the Yin-Yang  orbit are derived from numerical searches for periodic solutions
in the three-body problem, and they can be expressed in any consistent set of units. So while G=1 simplifies the 
code, it does not limit the generality of the solution as long as the initial conditions are correctly scaled.
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
G  = 1.0
dt = 0.0001
t_finish = 17.33   # one full period

# Initial conditions for Yin-Yang orbit 
masses   = [1.0, 1.0, 1.0]
n_bodies = len(masses)

positions = [
    np.array([-1.0, 0.0]),
    np.array([ 1.0, 0.0]),
    np.array([ 0.0, 0.0])
]

velocities = [
    np.array([ 0.46444,  0.39606]),
    np.array([ 0.46444,  0.39606]),
    np.array([-0.92888, -0.79212])
]

t_finish = 59.58  # one period


# storage for trajectories
trajectories = [[] for _ in range(n_bodies)]
for i in range(n_bodies):
    trajectories[i].append(positions[i].copy())

# acceleration function
def accelerations(pos):
    acc = [np.zeros(2) for _ in range(n_bodies)] # initialize acceleration list with zero vectors
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                r_vec = pos[j] - pos[i] # vector from body i to j
                r     = np.linalg.norm(r_vec)   # distance between bodies i and j
                acc[i] += G * masses[j] * r_vec / r**3  # Newton's law of gravitation for acceleration
    return acc  


# Energy Function
def compute_total_energy(pos, vel):
    """Compute total energy: kinetic + potential"""
    KE = sum(0.5 * masses[i] * np.dot(vel[i], vel[i]) for i in range(n_bodies)) # Kinetic Energy term 2 in equation 4
    PE = 0.0
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies): # Loop through unique pairs to avoid double counting
            r = np.linalg.norm(pos[j] - pos[i]) # distance between bodies i and j
            PE -= G * masses[i] * masses[j] / r # Potential Energy term 1 in equation 4
    return KE + PE


energy_time = []
total_energy = []
energy_sample_interval = 100  # record energy every N steps (keeps array manageable)
# initial acceleration
acc  = accelerations(positions)
time = 0.0
step = 0

# Simulation loop
while time <= t_finish:
    # Half-step velocity
    for i in range(n_bodies):
        velocities[i] += 0.5 * dt * acc[i]

    # Full-step position
    for i in range(n_bodies):
        positions[i] += dt * velocities[i]

    # New acceleration
    new_acc = accelerations(positions)

    # Second half-step velocity
    for i in range(n_bodies):
        velocities[i] += 0.5 * dt * new_acc[i]

    # Store trajectory
    for i in range(n_bodies):
        trajectories[i].append(positions[i].copy())

    # Energy (sampled to save memory)
    if step % energy_sample_interval == 0:
        E = compute_total_energy(positions, velocities)
        total_energy.append(E)
        energy_time.append(time + dt)

    time += dt
    acc = new_acc

# convert to arrays for plotting
traj = [np.array(t) for t in trajectories]

# Plots
colors  = ['blue', 'orange', 'purple']
labels  = ['Body 1', 'Body 2', 'Body 3']
colors2 = ['blue', 'orange', 'black']
shapes  = ['o', 'd', 's']

ARROW_LEN    = 0.15 # length of arrow to indicate direction of motion
ARROW_OFFSET = 0.1 # gap between arrowhead and final position (set to 0 for no gap)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
# Plot trajectories
for i in range(n_bodies):
    ax1.plot(traj[i][:, 0], traj[i][:, 1],
             color=colors2[i], lw=0.8, alpha=0.9, label=labels[i])

    ax1.scatter(traj[i][0, 0], traj[i][0, 1],
                color=colors[i], s=120, zorder=5,
                edgecolors='black', linewidths=0.8, marker=shapes[i],
                label='_nolegend_')

    ax1.scatter(traj[i][-1, 0], traj[i][-1, 1],
                color=colors[i], s=120, zorder=6,
                edgecolors='black', linewidths=0.8, marker=shapes[i])

    end_i   = len(traj[i]) - 1
    start_i = max(0, end_i - 40)

    tip       = traj[i][end_i,   :2]
    tail      = traj[i][start_i, :2]
    direction = tip - tail
    norm      = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm

    arrow_tip  = tip - direction * ARROW_OFFSET # position of arrow tip, offset from final point
    arrow_tail = arrow_tip - direction * ARROW_LEN # position of arrow tail, further back along the trajectory

# arrow at end of trajectory to show direction of motion
    ax1.annotate("",
        xy=arrow_tip, xytext=arrow_tail,
        arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=2.5, mutation_scale=20),
        zorder=3,
    )

ax1.set_xlabel("x (AU)", fontsize=13)
ax1.set_ylabel("y (AU)", fontsize=13)
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.4)

# Energy plot
ax2.plot(energy_time, total_energy, 'b-', lw=0.8)
ax2.set_xlabel("Time", fontsize=13)
ax2.set_ylabel("Relative Energy Change [(E-E0)/E0]", fontsize=13)
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
