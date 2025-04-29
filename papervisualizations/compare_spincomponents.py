import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mujoco
import random

random.seed(42)
np.random.seed(42)

from helper import cam2img, world2cam
from helper import HEIGHT, WIDTH, FPS
from helper import TABLE_LENGTH, TABLE_WIDTH, TABLE_HEIGHT
from helper import table_connections, table_points
from helper import get_Mext
from simulation.mujocosimulation import CAMERA, TIMESTEP, MAX_TIME, fx, fy, c, u, r
from simulation.mujocosimulation import XML, _calc_cammatrices


def get_trajectory(w0):
    ''' Get the trajectory of the ball in world coordinates.
    Args:
        w0 (np.array): initial spin of the ball in world coordinates
    '''
    # initialize variables
    r0 = np.array([1.5, 0, 1.2])  # initial position
    v0 = np.array([-10, 0, 0])  # initial velocity

    # initialize the simulation
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[0:3] = r0
    data.qvel[0:3] = v0
    data.qvel[3:6] = w0
    mujoco.mj_step(model, data)

    # calculate trajectory
    positions = []
    times = []
    next_save_time = 0.
    while next_save_time < MAX_TIME:
        steps = round((next_save_time - data.time) / TIMESTEP)
        mujoco.mj_step(model, data, steps)

        # check if ball is out of bounds
        if abs(data.qpos[0]) > 4 or abs(data.qpos[1]) > 2 or data.qpos[2] < 0.5:
            break

        positions.append(data.qpos[0:3].copy())
        times.append(data.time)
        next_save_time += 1 / FPS

    return np.array(positions)


def plot_trajectory2D(r_img, table_img, color='r'):
    '''Plot the image coordinates + plot the table points in the image plane. Plotting/saving has to be done manually
    -> We can plot multiple things in one plot
    Args:
        r_img (np.array): ball trajectory in image coordinates
        table_img (np.array): table points in image coordinates
        color (str): color of the trajectory. Format like in matplotlib
    '''
    # draw lines between the table points
    for connection in table_connections:
        plt.plot(table_img[connection, 0], table_img[connection, 1], 'k')
    # draw ground truth trajectory
    plt.plot(r_img[:, 0], r_img[:, 1], color)
    plt.scatter(r_img[0, 0], r_img[0, 1], c='g', label='Start')


def compare_trajectories():
    '''Compare the ball trajectory in image coordinates.
    '''
    ws = [np.array([0, 0, 0]), np.array([-100, 0, 0]), np.array([0, -100, 0]), np.array([0, 0, -100])]
    colors = ['r', 'b', 'g', 'y']
    labels = ['no spin', r'$w_{\tilde{x}}=-100\,\mathrm{Hz}$', r'$w_{\tilde{y}}=-100\,\mathrm{Hz}$', r'$w_{\tilde{z}}=-100\,\mathrm{Hz}$']
    linestyles = ['-', '--', '--', '--']
    f = np.cross(u, r)  # forward vector
    Mext = get_Mext(c, f, r)
    Mint = np.array([[fx, 0, (WIDTH-1)/2], [0, fy, (HEIGHT-1)/2], [0, 0, 1]])
    table_img = cam2img(world2cam(table_points, Mext), Mint)

    # 2D plot in image space
    #plt.title('Ball trajectory in image coordinates for different spins')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, WIDTH)
    plt.ylim(HEIGHT, 0)
    # draw lines between the table points
    for connection in table_connections:
        plt.plot(table_img[connection, 0], table_img[connection, 1], 'k')
    for w, color, label, linestyle in zip(ws, colors, labels, linestyles):
        r_world = get_trajectory(w)
        r_img = cam2img(world2cam(r_world, Mext), Mint)
        plt.plot(r_img[:, 0], r_img[:, 1], c=color, label=label, linestyle=linestyle)
        #plt.scatter(r_img[0, 0], r_img[0, 1], c='g', label='Start')
        plt.xlim(700, 2400)
        plt.ylim(1100, 350)
    # set aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    # Save image with high resolution
    #plt.savefig('trajectory_spincomponents.png', dpi=300)
    plt.show()

    # 3D plot in world space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Ball trajectory in world coordinates for different spins')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-4, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0.4, 1.5)
    for w, color, label in zip(ws, colors, labels):
        r_world = get_trajectory(w)
        ax.plot(r_world[:, 0], r_world[:, 1], r_world[:, 2], c=color, label=label)
        ax.scatter(r_world[0, 0], r_world[0, 1], r_world[0, 2], c='g', label='Start')
    # plot the table connections in 3D
    for connection in table_connections:
        ax.plot(table_points[connection, 0], table_points[connection, 1], table_points[connection, 2], 'k')
    ax.legend()
    plt.show()

    # calculate distances between trajectories
    for i, w in enumerate(ws[1:]):
        r0 = get_trajectory(ws[0])
        r1 = get_trajectory(w)
        min_length = min(len(r0), len(r1))
        dist = np.mean(np.linalg.norm(r0[:min_length, :] - r1[:min_length, :], axis=-1))
        print(f'3D Distance between {labels[0]} and {labels[i+1]}: {dist}m')
        r0_img = cam2img(world2cam(r0, Mext), Mint)
        r1_img = cam2img(world2cam(r1, Mext), Mint)
        dist_img = np.mean(np.linalg.norm(r0_img[:min_length, :] - r1_img[:min_length, :], axis=-1))
        print(f'2D Distance between {labels[0]} and {labels[i+1]}: {dist_img}pxl')



if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    compare_trajectories()


