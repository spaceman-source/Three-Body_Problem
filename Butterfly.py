 

"""
James Ward
3/31/26
Butterfly I Orbit Simulation (Three-Body Problem)
Natural units: G = 1, masses = 1, distances and time dimensionless.
Initial conditions: Šuvakov & Dmitrašinović (PRL 110, 114301, 2013)
Period T ≈ 6.2356
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
G = 1.0       # Gravitational constant 
dt = 0.0001   # Time step
t_finish =  6.2356

# Initial conditions for Butterfly I orbit 
masses = [1.0, 1.0, 1.0]   # Equal masses
n_bodies = len(masses)

# Initial conditions from Šuvakov & Dmitrašinović 2013, bodies start collinear on x-axis.
positions = [
    np.array([-0.99902,  0.0]),
    np.array([ 0.99902,  0.0]),
    np.array([ 0.0,      0.0])
]

velocities = [
    np.array([ 0.347111,  0.532728]),
    np.array([ 0.347111,  0.532728]),
    np.array([-0.694222, -1.065456])
]

# Trajectory Lists
trajectories = [[] for _ in range(n_bodies)]
for i in range(n_bodies):
    trajectories[i].append(positions[i].copy())

# Energy Lists
energy_time = []
total_energy = []

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

# Initial energy
E0 = compute_total_energy(positions, velocities)
total_energy.append(E0)
energy_time.append(0.0)

# Acceleration Function
def accelerations(pos):
    acc = [np.zeros(2) for _ in range(n_bodies)]    # Initialize acceleration array
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                r_vec = pos[j] - pos[i] # Vector from body i to j
                r = np.linalg.norm(r_vec) # distance between bodies i and j
                acc[i] += G * masses[j] * r_vec / r**3 # Newton's law of gravitation for acceleration
    return acc

# Initial acceleration
acc = accelerations(positions) # compute initial acceleration based on initial positions

# 
time = 0.0
energy_sample_interval = 100   # record energy every N steps (keeps array manageable)
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
    step += 1


# Convert trajectory to an array
traj = [np.array(t) for t in trajectories]

# Relative energy change for energy conservation check
energy_arr = np.array(total_energy)
rel_energy_err = (energy_arr - E0) / abs(E0)

# Plotting settings
labels  = ['Body 1', 'Body 2', 'Body 3']
colors  = ['royalblue', 'darkorange', 'purple']
colors2 = ['royalblue', 'darkorange', 'black']
shapes  = ['o', 'd', 's']

ARROW_LEN    = 0.15 # length of arrow to indicate direction of motion
ARROW_OFFSET = 0.0 # gap between arrowhead and final position (set to 0 for no gap)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Trajectory plot
for i in range(n_bodies):
    ax1.plot(traj[i][:, 0], traj[i][:, 1],
             color=colors2[i], lw=0.6, alpha=0.8, label=labels[i])

    # Start marker
    ax1.scatter(traj[i][0, 0], traj[i][0, 1],
                color=colors[i], s=120, zorder=5,
                edgecolors='black', linewidths=0.8, marker=shapes[i],
                label=f'{labels[i]} Start')

    

    # Arrow at end
    end_i   = len(traj[i]) - 1
    start_i = max(0, end_i - 40)

    tip       = traj[i][end_i,   :2]
    tail      = traj[i][start_i, :2]
    direction = tip - tail
    norm      = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm

    arrow_tip  = tip - direction * ARROW_OFFSET
    arrow_tail = arrow_tip - direction * ARROW_LEN

    ax1.annotate("",
        xy=arrow_tip, xytext=arrow_tail,
        arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=2.5, mutation_scale=20),
        zorder=3,
    )

ax1.set_xlabel("x (AU)", fontsize=13)
ax1.set_ylabel("y (AU)", fontsize=13)
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.4)


# Energy conservation plot
ax2.plot(energy_time, rel_energy_err, 'b-', lw=0.8)
ax2.set_xlabel("Time", fontsize=13)
ax2.set_ylabel("Relative energy Change  [(E - E₀) / E₀]", fontsize=13)
ax2.grid(True, linestyle='--', alpha=0.4)


plt.tight_layout()
plt.show()
