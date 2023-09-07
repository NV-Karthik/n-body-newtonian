import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# definitions

# m vector contains masses of all particles
# pos vector contains space-coordinates as Nx3 2d-array, r[i] = [x, y, z])
# vel vector contains velocities in Nx3 2d array, vel[i] = [v_x, v_y, v_z]
# acc vector contains accelerations in Nx3 array, acc[i] = [a_x, a_y, a_z]


# vectorized subroutines
def getAcceleration(masses, positions, G_constant):
    # masses is a Nx1 array
    # define a constant param to avoid division by zero
    k = 0.01

    x = positions[:, 0:1] # we need these arrays to be 2d for the
    y = positions[:, 1:2] # following numpy trick to work
    z = positions[:, 2:3]

    # get pairwise distances
    x_dist = x.T - x
    y_dist = y.T - y
    z_dist = z.T - z

    inv_dist3 = ( (x_dist**2) + (y_dist**2) + (z_dist**2) + k)**(-1.5) # NxN array
    # inv_dist3 = (x_dist**2) + (y_dist**2) + (z_dist**2) + k
    # inv_dist3[inv_dist3>0] = inv_dist3[inv_dist3>0]**(-1.5)

    # NxN and Nx1 matmul gives Nx1 array, ith row for ith particle
    ax = G_constant * (x_dist * inv_dist3) @ masses
    ay = G_constant * (y_dist * inv_dist3) @ masses
    az = G_constant * (z_dist * inv_dist3) @ masses

    # pack together into one Nx3 2d-array
    a = np.hstack((ax, ay, az))

    return a

def getEnergies(masses, positions, velocities, G_constant):

    # Kinetic energy - have to sum over arrays twice
    KE = 0.5 * np.sum(np.sum(masses * (velocities**2)))

    # Potential energy
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)

    inv_r[inv_r>0] = inv_r[inv_r>0]**-1 # remove 0 distance entries

    PE = G_constant * np.sum(np.sum(np.triu(-(masses*masses.T)*inv_r,1)))

    return KE, PE



# initialize parameters

# time params
t_init = 0
t_final = 20 # sec
dt = 0.01 # time step
num_steps = int(np.ceil(t_final/dt)) # number of time steps / iterations

# system conditions

N = 100 # number of particles
dims = 3 # num of dimensions
G = 1.0 # newtons gravitational constant, (assumed 1 for numerical accuracy)
total_mass = 20 # total mass of all particles


pos = np.random.randn(N, dims) # define random set of points in space
vel = np.random.randn(N, dims) # assign random velocities to each point
mas = np.ones((N, 1)) * (total_mass/N) # all masses are 1unit, create specifically a column vector only

# Convert to Center-of-Mass frame
vel -= np.mean(mas * vel,0) / np.mean(mas)

# get first set of accelerations
acc = getAcceleration(mas, pos, G)

# preallocate simulation data arrays
pos_data = np.zeros((N, 3, num_steps+1))
pos_data[:,:,0] = pos

# similarly for energy todo
ke_data = np.zeros(num_steps+1)
pe_data = np.zeros(num_steps+1)

ke, pe = getEnergies(mas, pos, vel, G)

# simulation loop
for i in range(num_steps):
    # first half kick
    vel += acc * (0.5*dt)

    # drift
    pos += vel * dt

    # update acc
    acc = getAcceleration(mas, pos, G)

    # 2nd kick
    vel += acc * (0.5*dt)

    # get energies
    KE, PE = getEnergies(mas, pos, vel, G)

    # save position, energy data
    pos_data[:, :, i+1] = pos
    ke_data[i+1] = KE
    pe_data[i+1] = PE



# create animation of n body motion
fig, ax = plt.subplots(figsize=(8,8))

def update(num):
    # update coordinates
    x_i = pos_data[:, 0, num]
    y_i = pos_data[:, 1, num]
    z_i = pos_data[:, 2, num]
    
    ax.cla()
    # ax.set_title('Delaunay Triangulation')
    xx = pos_data[:,0,max(num-50,0):num+1]
    yy = pos_data[:,1,max(num-50,0):num+1]
    ax.scatter(xx,yy,s=1,color=[.7,.7,1])
    ax.scatter(x_i, y_i, color='black')
    ax.set_aspect('equal')
    # ax.set(xlim=(-2,2),ylim=(-2,2))
    ax.set_axis_off()
    return fig,

ani = animation.FuncAnimation(fig, update, num_steps, interval= dt*1000, blit=True)
ani.save('100-body.mp4', writer="ffmpeg",dpi=100)

# create plot of energies wrt time

plt.clf() # clear this figure

t_vals = np.arange(num_steps+1)*dt
plt.scatter(t_vals, ke_data, color='red', label='KE', s=2.5)
plt.scatter(t_vals, pe_data, color='blue', label='PE', s=2.5)
plt.scatter(t_vals, ke_data + pe_data, color='black', label='Total Energy', s=2.5)
plt.xlabel('time')
plt.ylabel('energy')
plt.legend(loc='upper right')

plt.savefig('energy_plots_n100.jpg', bbox_inches='tight', dpi=100)

