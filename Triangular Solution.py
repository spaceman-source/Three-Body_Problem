
"""
    James Ward
    This Program examines the three-body problem specifically the triangular solution
    Masses are scaled in solar masses and distances are scaled in Astronomical Units.
"""
# Import libraries
import matplotlib.pyplot as plt # Need this for plotting
import matplotlib.style 
import numpy as np # Need this for math



# Gravitational constant in terms of AU, Derived from Keplers 3rd Law, Equation 1 from poster
G = 4 * np.pi**2


# Time step 
dt = 1/1000


# Total simulation time in years 
t_finish = 5
time = 0.0

# triangular has unequal masses and triangular startup
masses = [100, 5.0, 1.0]   # tweak these to change radii
M     = sum(masses)
a     = 2               # side length in AU



# Setup for equilateral triangal with center of mass at center
angles_c  = [np.radians(90), np.radians(210), np.radians(330)]
R_c       = a / np.sqrt(3)
r_centroid = [R_c * np.array([np.cos(ang), np.sin(ang)]) for ang in angles_c]

# Find center of mass
com_offset = sum(masses[i] * r_centroid[i] for i in range(3)) / M

# Shift every body so COM sits at origin
positions_initial = [r_centroid[i] - com_offset for i in range(3)]
positions = [r_centroid[i] - com_offset for i in range(3)]



#  Same angular velocity for all bodies
omega = np.sqrt(G * M / a**3)
velocities = [np.array([-omega * p[1], omega * p[0]]) for p in positions]


# number of bodies equals how many values are in masses
n_bodies = len(masses)


# Store trajectories
trajectories = [[] for _ in range(n_bodies)]

# Append Positions to get trajectory
for i in range(n_bodies):
    trajectories[i].append(positions[i].copy())

# Energy Conservation Check, use this to see if energy is being conserved

energy_time = []      # time values for energy plot
total_energy = []     # total energy at each time


# Function for Energy Conservation, Equation 4 is just KE + PE
def total_energylist(pos, vel):
    """Compute total mechanical energy (kinetic + potential)"""
    KE = 0.0
    for i in range(n_bodies):
        KE += 0.5 * masses[i] * np.dot(vel[i], vel[i]) # Kinetic energy term 1 in equation 4
    PE = 0.0
    for i in range(n_bodies):
        for j in range(i+1, n_bodies):
            r_vec = pos[j] - pos[i] # position vector
            r = np.linalg.norm(r_vec) # magnitude of position vector
            PE -= G * masses[i] * masses[j] / r # Potential Energy equation term 4 in equation
    return KE + PE

E0 = total_energylist(positions, velocities) # Total Energy
energy_time.append(time)  # store time associated with energy
total_energy.append(E0) # Store total energy of system



# Function to compute accelerations from current positions
def accelerations(pos):
    acc = [np.zeros(2) for _ in range(n_bodies)] # start with no acceleration, sets initial accelerations to zero
    for i in range(n_bodies):   #  Loop through each body in system i is current body computing acceleration for
        for j in range(n_bodies):
            if i == j: # can't exert force on itself, make sure its not trying to 
                continue
            r_vec = pos[j] - pos[i] # computes vector pointing towards j
            r = np.linalg.norm(r_vec) # distance between i and j, computes magnitude
            acc[i] = acc[i] + G * masses[j] * r_vec / r**3 # Acceleration equation 3 from poster
    return acc

# Initial accelerations 
acc = accelerations(positions) # compute initial acceleration using acceleration function


# Simulation loop Equations 5 and 6
# This code uses the Leapfrog method,  which uses a half step for velocity and a full step for
# position, this is a symplectic integrator that helps with energy conservation in long term simulations
while time <= t_finish:
    #  Half velocity using current acceleration
    for i in range(n_bodies):           
        velocities[i] = velocities[i] + 0.5 * dt * acc[i] # v(t = 1/2dt)

    # Full step position using the half step
    for i in range(n_bodies):
        positions[i] = positions[i] + dt * velocities[i] # x(t + dt) 

    # Compute new acceleration using new positions
    new_acc = accelerations(positions)

    # Second half step using new acceleration
    for i in range(n_bodies):
        velocities[i] = velocities[i] + 0.5 * dt * new_acc[i] # v(t + dt)

    # Use trajectories array to store the positions
    for i in range(n_bodies):
        trajectories[i].append(positions[i].copy())


    # Energy conservation
    E = total_energylist(positions, velocities)
    energy_time.append(time + dt)   # time after this step
    total_energy.append(E)
    

    # Update time and accelerations for next step
    time = time + dt
    acc = new_acc

# Convert trajectories to arrays for plotting
traj = [np.array(t) for t in trajectories]


# Combined energy and trajectory plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Colors and Labels of bodies
colors = ['blue', 'orange', 'purple']
labels = ['Body 1 ', 'Body 2 ', 'Body 3 ']

# Left trajectory plot 
for i in range(n_bodies):
    Ri = np.linalg.norm(positions_initial[i]) # Distance from origin, used to plot initial circles around each body

    ax1.plot(traj[i][:, 0], traj[i][:, 1], # plots trajectory
             linewidth=0.8, alpha=0.6, color=colors[i])

 # Add direction arrows
    j = len(traj[i]) // 3 # 1/3 of the way through trajectory 
    dx = traj[i][j + 1, 0] - traj[i][j, 0]
    dy = traj[i][j + 1, 1] - traj[i][j, 1]
    ax1.arrow(traj[i][j, 0], traj[i][j, 1], dx, dy,
              head_width=0.04, color=colors[i], length_includes_head=True)

    ax1.plot(*positions_initial[i], 'o', markersize=8,
             color=colors[i], label=f'{labels[i]}  $R={Ri:.2f}$ AU')


# Plot the  triangle
x_triangle = [positions_initial[i][0] for i in range(n_bodies)]
y_triangle = [positions_initial[i][1] for i in range(n_bodies)]
x_triangle.append(x_triangle[0])
y_triangle.append(y_triangle[0])
ax1.plot(x_triangle, y_triangle, 'r:', linewidth=2)

# Setup for Legend
ax1.set_xlabel('x (AU)', fontsize=13)
ax1.set_ylabel('y (AU)', fontsize=13)
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(fontsize=12, loc = 'upper right')

# Energy Drift Plot
energy_time = np.array(energy_time)
total_energy = np.array(total_energy)
relative_change = (total_energy - E0) / E0

ax2.plot(energy_time, relative_change, 'b-')
ax2.set_xlabel('Time (years)', fontsize=13)
ax2.set_ylabel('Relative Energy Change', fontsize=13)
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"Initial energy: {E0:.6e} (solar mass * AU² / yr²)")
print(f"Final energy:   {total_energy[-1]:.6e}")
print(f"Relative change: {relative_change[-1]:.3e}")
print(f"positions_initial: {positions_initial}")