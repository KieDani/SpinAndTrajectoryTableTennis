if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate inference dataset')
    parser.add_argument('--num_trajectories', type=int, default=50000, help='Number of trajectories to generate')
    parser.add_argument('--folder', type=str, default='data50000', help='Folder to save the dataset')
    parser.add_argument('--num_processes', type=int, default=128, help='Number of cpu processes to use')
    args = parser.parse_args()
import mujoco
import mujoco_viewer
import numpy as np
import random
import math
import os
import tqdm
from multiprocessing import Pool
import einops as eo

from helper import cam2img, world2cam
from helper import HEIGHT, WIDTH, FPS
from helper import TABLE_LENGTH, TABLE_WIDTH, TABLE_HEIGHT

CAMERA = 'main'

# MuJoCo stuff
TIMESTEP = 0.001
MAX_TIME = 0.8

fx, fy = 2796, 2679
c = np.array([0.04381194, 8.92938715, 5.40070126])
u = np.array([7.81340900e-04, -4.33644716e-01, 9.01083598e-01])
r = np.array([-0.99998599, 0.00437903, 0.0029745])


XML = f"""
<mujoco>
  <option cone="elliptic" gravity="0 0 -9.81" integrator="implicit" timestep="{TIMESTEP}" density="1.225" viscosity="0.000018" />
  <asset>
    <material name="ball_material" reflectance="0" rgba="1 1 1 1"/>
    <texture name="table_texture" type="cube" filefront="simulation/table.png" fileup="simulation/black.png" filedown="simulation/black.png" fileleft="simulation/black.png" fileright="simulation/black.png" />
    <material name="table_material" reflectance=".2" texture="table_texture" texuniform="false"/>
    <texture name="net_texture" type="cube" fileleft="simulation/net.png" fileright="simulation/net.png" />
    <material name="net_material" reflectance="0" texture="net_texture" texuniform="false" texrepeat="1 1" />
  </asset>
  <visual>
    <global offheight="{HEIGHT}" offwidth="{WIDTH}"/>
  </visual>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="-2 -2 10" dir="0.1 0.1 -1"/>
    <geom type="plane" size="800 800 0.1" rgba="0.6 0.6 0.6 1" pos="0 0 0" material="ball_material"/>
    <body name="ball_body" pos="0 0 1.2">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size=".02" material="ball_material" mass=".0027" fluidshape="ellipsoid" fluidcoef="0.235 0.25 0.0 1.0 1.0"/> 
    </body>
    <geom name="table_geom" type="box" pos="0 0 {TABLE_HEIGHT/2}" size="{TABLE_LENGTH/2} {TABLE_WIDTH/2} {TABLE_HEIGHT/2}" material="table_material"/>
    <geom name="net_geom" type="box" pos="0 0 {TABLE_HEIGHT}" size="0.02 {TABLE_HEIGHT+0.1525} 0.1525" material="net_material" rgba="1 1 1 0.6" />
    <!-- Define the camera -->
    <camera name="main" 
            focal="{fx/WIDTH} {fy/HEIGHT}"
            resolution="{WIDTH} {HEIGHT}"
            sensorsize="1 1"
            pos="{c[0]} {c[1]} {c[2]}" 
            mode="fixed" 
            xyaxes="{r[0]} {r[1]} {r[2]} {u[0]} {u[1]} {u[2]}"/>
  </worldbody>
  <default>
    <pair solref="-1000000 -17" solreffriction="-0.0 -200.0" friction="0.1 0.1 0.005 0.0001 0.0001" solimp="0.98 0.99 0.001 0.5 2"/>
  </default>

  <contact>
    <pair geom1="ball_geom" geom2="table_geom"/>
    <pair geom1="ball_geom" geom2="net_geom"/>
  </contact>
</mujoco>
"""


def _init_simulation(seed):
    # set seeds
    random.seed(seed)
    np.random.seed(seed)

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # define initial position and velocity
    # initial position:  0.2 < x < 4, -2 < y < 2, 0.5 < z < 1.8
    r = np.empty(3, dtype=np.float64)
    r[0] = random.uniform(0.2, 4)
    r[1] = random.uniform(-2, 2)
    if abs(r[0]) < TABLE_LENGTH/2 and abs(r[1]) < TABLE_WIDTH/2:  # ball should not penetrate the table
        r[2] = random.uniform(0.8, 1.8)
    else:
        r[2] = random.uniform(0.5, 1.8)
    # initial velocity: 3 < |v| < 35, 110° < phi < 250°, 10° < theta < 170°
    v = np.empty(3, dtype=np.float64)
    speed = random.uniform(3, 35)
    phi = random.uniform(np.deg2rad(110), np.deg2rad(250))
    theta = random.uniform(np.deg2rad(10), np.deg2rad(170))
    v[0] = speed * math.sin(theta) * math.cos(phi)
    v[1] = speed * math.sin(theta) * math.sin(phi)
    v[2] = speed * math.cos(theta)
    # initial angular velocity: 0 < |w| < 200
    w = np.zeros(3, dtype=np.float64)
    speed = random.uniform(0, 200)
    phi = random.uniform(0, 2 * math.pi)
    theta = random.uniform(0, math.pi)
    w[0] = speed * math.sin(theta) * math.cos(phi)
    w[1] = speed * math.sin(theta) * math.sin(phi)
    w[2] = speed * math.cos(theta)

    # set initial position and velocity
    data.qpos[0:3] = r
    data.qvel[0:3] = v
    data.qvel[3:6] = w

    return model, data


def _count_hits(positions):
    hits_own = []
    hits_opponent = []
    hits_ground = []
    threshold = TABLE_HEIGHT + 0.04
    binary_mask_z = np.array([pos[2] < threshold for pos in positions])  # True if ball is below the threshold
    binary_mask_y = np.array([abs(pos[1]) < TABLE_WIDTH/2 for pos in positions])  # True if ball is within the table
    binary_mask_x_opponent = np.array([-0.01 > pos[0] > -TABLE_LENGTH/2 for pos in positions])  # True if ball is within the table on the opposite side
    binary_mask_x_own = np.array([TABLE_LENGTH/2 > pos[0] > 0.01 for pos in positions])  # True if ball is within the table on the own side
    binary_mask_opponent = binary_mask_z & binary_mask_y & binary_mask_x_opponent
    binary_mask_own = binary_mask_z & binary_mask_y & binary_mask_x_own
    binary_mask_ground = np.array([pos[2] <= 0.05 for pos in positions])  # True if ball hits the ground
    positions = np.array(positions)
    start, end = None, None
    for i, b in enumerate(binary_mask_opponent):
        if i == 0 and b:
            start = i
        elif b and not binary_mask_opponent[i - 1]:
            start = i
        if not b and binary_mask_opponent[i - 1] and i != 0:
            end = i - 1
            hits_opponent.append(0.75 * (end + start) / 2 / FPS + 0.25 * (np.argmin(positions[start:end+1, 2]) + start) / FPS)
    start, end = None, None
    for i, b in enumerate(binary_mask_own):
        if i == 0 and b:
            start = i
        elif b and not binary_mask_own[i - 1]:
            start = i
        if not b and binary_mask_own[i - 1] and i != 0:
            end = i - 1
            hits_own.append(0.75 * (end + start) / 2 / FPS + 0.25 * (np.argmin(positions[start:end+1, 2]) + start) / FPS)
    start, end = None, None
    for i, b in enumerate(binary_mask_ground):
        if i == 0 and b:
            start = i
        elif b and not binary_mask_ground[i - 1]:
            start = i
        if not b and binary_mask_ground[i - 1] and i != 0:
            end = i - 1
            hits_ground.append(0.75 * (end + start) / 2 / FPS + 0.25 * (np.argmin(positions[start:end+1, 2]) + start) / FPS)
    return hits_opponent, hits_own, hits_ground


def get_valid_trajectories_pool(seeds):
    valid_solutions = []
    valid_seeds = []
    for seed in seeds:
        # initialize the simulation
        model, data = _init_simulation(seed)
        # renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        mujoco.mj_step(model, data)
        # renderer.update_scene(data, camera=CAMERA)

        positions, velocities, rotations = [], [], []
        times = []
        next_save_time = 0.
        while next_save_time < MAX_TIME:
            steps = round((next_save_time - data.time) / TIMESTEP)
            mujoco.mj_step(model, data, steps)

            # check if ball is out of bounds
            if abs(data.qpos[0]) > 4 or abs(data.qpos[1]) > 2 or data.qpos[2] < 0.5:
                break
            # check if ball is still in the image plane
            # renderer.update_scene(data, camera=CAMERA)
            ex_mat, in_mat = _calc_cammatrices(data, camera_name=CAMERA)
            r_cam = world2cam(data.qpos[0:3].copy(), ex_mat)
            r_img = cam2img(r_cam, in_mat[:3, :3])
            if not np.all((r_img >= 0) & (r_img < np.array([WIDTH, HEIGHT]))):
                break

            positions.append(data.qpos[0:3].copy())
            velocities.append(data.qvel[0:3].copy())
            rotations.append(data.qvel[3:6].copy())
            times.append(next_save_time)
            next_save_time += 1 / FPS

        # ensure minimum length of 7 frames
        if len(positions) < 7:
            continue
        # check if ball hit the table exactly once at the opponents side -> Neither final nor first trajectory
        hits_opponent, hits_own, hits_ground = _count_hits(positions)
        if len(hits_opponent) == 1 and len(hits_own) == 0 and len(hits_ground) == 0:
            valid_solutions.append((np.array(positions), np.array(velocities), np.array(rotations), np.array(times), np.array(hits_opponent)))
            valid_seeds.append(seed)
            print('Found solution with seed ', seed)
    return (valid_solutions, valid_seeds)


def get_valid_trajectories(num_trajectories, num_processes):
    valid_solutions = []
    valid_seeds = []
    current_seed = 0
    batch_size = min(1024, num_trajectories)
    while len(valid_solutions) < num_trajectories:
        seeds = [list(range(current_seed + j, current_seed + batch_size, num_processes)) for j in range(num_processes)]
        with Pool(num_processes) as p:
            results = p.map(get_valid_trajectories_pool, seeds)
        for result_p in results:
            vsolutions, vseeds = result_p[0], result_p[1]
            valid_solutions.extend(vsolutions)
            valid_seeds.extend(vseeds)
        current_seed += batch_size
    valid_solutions = valid_solutions[:num_trajectories]
    valid_seeds = valid_seeds[:num_trajectories]


    print(f"Found {len(valid_solutions)} valid solutions.")
    print(f"Seeds: {valid_seeds}")

    return valid_solutions, valid_seeds


def _calc_cammatrices(data, camera_name):
    camera_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    weird_R = eo.rearrange(data.cam_xmat[camera_id], '(i j) -> i j', i=3, j=3).T
    R = np.eye(3)
    R[0, :] = weird_R[0, :]
    R[1, :] = - weird_R[1, :]
    R[2, :] = - weird_R[2, :]
    cam_pos = data.cam_xpos[camera_id]
    t = -np.dot(R, cam_pos)

    ex_mat = np.eye(4)
    ex_mat[:3, :3] = R
    ex_mat[:3, 3] = t

    fx = data.model.cam_intrinsic[camera_id][0] / data.model.cam_sensorsize[camera_id][0] * data.model.cam_resolution[camera_id][0]
    fy = data.model.cam_intrinsic[camera_id][1] / data.model.cam_sensorsize[camera_id][1] * data.model.cam_resolution[camera_id][1]
    cx = (WIDTH - 1) / 2
    cy = (HEIGHT - 1) / 2
    in_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1], [0, 0, 0]])

    return ex_mat, in_mat


def generate_dataset(path, num_trajectories=200, num_processes=8):
    '''Generates Dataset. No images are saved.'''
    # get valid trajectories (inside reasonable bounds and inside the image)
    valid_solutions, valid_seeds = get_valid_trajectories(num_trajectories, num_processes)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    for i, (solution, seed) in tqdm.tqdm(enumerate(zip(valid_solutions, valid_seeds))):
        # initialize the simulation
        model, data = _init_simulation(seed)
        # renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        mujoco.mj_step(model, data)
        # renderer.update_scene(data, camera=CAMERA)
        # do simulation
        positions = []
        velocities = []
        times = []
        ex_mats = []
        in_mats = []
        rotations = []
        next_save_time = 0.
        while next_save_time < MAX_TIME:
            steps = round((next_save_time - data.time) / TIMESTEP)

            mujoco.mj_step(model, data, steps)
            # check if ball is out of bounds
            if abs(data.qpos[0]) > 4 or abs(data.qpos[1]) > 2 or data.qpos[2] < 0.5:
                break
            # check if ball is still in the image plane
            # renderer.update_scene(data, camera=CAMERA)
            ex_mat, in_mat = _calc_cammatrices(data, camera_name=CAMERA)
            r_cam = world2cam(data.qpos[0:3].copy(), ex_mat)
            r_img = cam2img(r_cam, in_mat[:3, :3])
            if not np.all((r_img >= 0) & (r_img < np.array([WIDTH, HEIGHT]))):
                break
            positions.append(data.qpos[0:3].copy())
            velocities.append(data.qvel[0:3].copy())
            times.append(next_save_time)
            # renderer.update_scene(data, camera=CAMERA)
            ex_mat, in_mat = _calc_cammatrices(data, camera_name=CAMERA)
            in_mat = in_mat[:3, :3]
            ex_mats.append(ex_mat)
            in_mats.append(in_mat)
            rotations.append(data.qvel[3:6].copy())
            next_save_time += 1 / FPS
        model, data = _init_simulation(seed)
        # renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        mujoco.mj_step(model, data)
        blur_positions = []
        blur_times = []
        ex_mats_blur = []
        in_mats_blur = []
        next_save_time_blur = 0.
        while next_save_time_blur < MAX_TIME:
            steps = round((next_save_time_blur - data.time) / TIMESTEP)
            mujoco.mj_step(model, data, steps)
            # check if ball is out of bounds
            if abs(data.qpos[0]) > 4 or abs(data.qpos[1]) > 2 or data.qpos[2] < 0.5:
                break
            # check if ball is still in the image plane
            # renderer.update_scene(data, camera=CAMERA)
            ex_mat, in_mat = _calc_cammatrices(data, camera_name=CAMERA)
            r_cam = world2cam(data.qpos[0:3].copy(), ex_mat)
            r_img = cam2img(r_cam, in_mat[:3, :3])
            if not np.all((r_img >= 0) & (r_img < np.array([WIDTH, HEIGHT]))):
                break
            blur_positions.append(data.qpos[0:3].copy())
            blur_times.append(next_save_time_blur)
            # renderer.update_scene(data, camera=CAMERA)
            ex_mat, in_mat = _calc_cammatrices(data, camera_name=CAMERA)
            in_mat = in_mat[:3, :3]
            ex_mats_blur.append(ex_mat)
            in_mats_blur.append(in_mat)
            next_save_time_blur += 1 / (FPS*10)


        hits_opponent, hits_own, hits_ground = _count_hits(positions)

        assert np.allclose(solution[0], positions), f"Position mismatch for trajectory {i}"
        assert np.allclose(solution[3], times), f"Time mismatch for trajectory {i}"

        # saving the trajectory
        save_path = os.path.join(path, f"trajectory_{i:04}")
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'Mext.npy'), np.array(ex_mats))
        np.save(os.path.join(save_path, 'Mint.npy'), np.array(in_mats))
        np.save(os.path.join(save_path, 'rotations.npy'), np.array(rotations))
        np.save(os.path.join(save_path, 'times.npy'), np.array(times))
        np.save(os.path.join(save_path, 'positions.npy'), np.array(positions))
        np.save(os.path.join(save_path, 'velocities.npy'), np.array(velocities))
        np.save(os.path.join(save_path, 'hits.npy'), np.array(hits_opponent)) # only save the hits at the opponents side because we only look at this special case at the moment
        np.save(os.path.join(save_path, 'blur_positions.npy'), np.array(blur_positions))
        np.save(os.path.join(save_path, 'blur_times.npy'), np.array(blur_times))
        np.save(os.path.join(save_path, 'blur_Mext.npy'), np.array(ex_mats_blur))
        np.save(os.path.join(save_path, 'blur_Mint.npy'), np.array(in_mats_blur))


def test_visualization():
    """Init simulation and show it with mujoco viewer"""
    model, data = _init_simulation(33)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.cam.distance = 5
    # simulate and render
    for _ in range(10000):
        if viewer.is_alive:
            mujoco.mj_step(model, data)
            if _ % 50 == 0:
                viewer.render()
        else:
            break
    # close
    viewer.close()


if __name__ == "__main__":
    from paths import data_path
    path = os.path.join(data_path, args.folder)
    # First run: xvfb-run -a -s "-screen 0 1400x900x24" bash
    generate_dataset(path, args.num_trajectories, num_processes=args.num_processes)

    pass