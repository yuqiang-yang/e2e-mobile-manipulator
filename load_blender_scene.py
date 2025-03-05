import blenderproc as bproc
import concurrent.futures
import numpy as np
import bpy
import torch
import argparse
import time
import re
import open3d as o3d

from typing import Tuple
from typing import List 

from blenderproc.python.types.MeshObjectUtility import MeshObject, create_primitive
from blenderproc.python.types.URDFUtility import URDFObject
from blenderproc.python.types.EntityUtility import Entity, convert_to_entity_subclass

#curobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenStatus,
    MotionGenResult
)

# region Global
#####################Global Variables#######################
NUM_ROBOTS = 2
ROBOT_SEED = 4
last_success_target = np.zeros(3)
success_result = None
curobo_state = np.zeros((NUM_ROBOTS * ROBOT_SEED, 12)) 
task_finish = True
cmd_idx = 0
cmd_trajs = None
cube_pos_history = np.zeros((2, 3))   
cmd_step_size = 1
############################################################

# region Function
def hide_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection:
        collection.hide_viewport = True
        collection.hide_render = True
        collection.hide_select = True

def hide_inertial():
    for obj in bpy.context.scene.objects:
        if "finger_inertial" in obj.name or "effector_inertial" in obj.name:
            obj.hide_viewport = True
            obj.hide_render = True
            
def get_collision_map(objs: list, room_center: np.ndarray, motion_gen, tensor_args, bbox_size: tuple = (20, 20), resolution: float = 0.1):

    # Define the bounding box based on room center and size
    x_min = room_center[0] - bbox_size[0] / 2
    x_max = room_center[0] + bbox_size[0] / 2
    y_min = room_center[1] - bbox_size[1] / 2
    y_max = room_center[1] + bbox_size[1] / 2

    # Generate a meshgrid with the given resolution
    x_coords = np.arange(x_min, x_max, resolution)
    y_coords = np.arange(y_min, y_max, resolution)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Flatten the grid for iteration
    x_flat = grid_x.flatten()
    y_flat = grid_y.flatten()

    # Initialize the collision map
    collision_map = []

    # Initial joint state for feasibility checks
    init_arm_pose = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0])
    check_state = np.zeros(10)
    check_state[2] = 0.0  # yaw
    check_state[3:] = init_arm_pose

        
    # Iterate over each grid point
    for x, y in zip(x_flat, y_flat):
        # Set the current position for feasibility check
        check_state[:2] = [x, y]
        check_state_curobo = JointState.from_position(
            tensor_args.to_device(check_state),
            joint_names=motion_gen.rollout_fn.joint_names
        )

        # Check if the state is feasible
        valid, status = motion_gen.check_start_state(check_state_curobo)
        if status != MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION and not valid:
            print(f"status {status}  {[x, y]}")
            
        feasible = bool(valid)

        # Append the result to the collision map
        collision_map.append((x, y, feasible))

    return collision_map

def obj_to_ply_with_collision_map(obj_file: str, ply_file: str, collision_map: List[Tuple[float, float, bool]], z_height: float = 0.2, num_samples: int = 100000):
    # Load the OBJ file using Open3D
    mesh = o3d.io.read_triangle_mesh(obj_file)
    
    # Uniformly sample points from the mesh surface
    sampled_points = mesh.sample_points_uniformly(number_of_points=num_samples)
    obstacle_vertices = np.asarray(sampled_points.points)
    obstacle_colors = np.tile([128, 128, 128], (obstacle_vertices.shape[0], 1))  # Gray color for obstacles

    # Create point clouds for feasible and infeasible regions
    feasible_points = []
    infeasible_points = []

    for x, y, feasible in collision_map:
        if feasible:
            feasible_points.append([x, y, z_height])
        else:
            infeasible_points.append([x, y, z_height])

    feasible_points = np.array(feasible_points) if feasible_points else np.empty((0, 3))
    infeasible_points = np.array(infeasible_points) if infeasible_points else np.empty((0, 3))

    feasible_colors = np.tile([0, 0, 255], (feasible_points.shape[0], 1))  # Blue color for feasible points
    infeasible_colors = np.tile([255, 0, 0], (infeasible_points.shape[0], 1))  # Red color for infeasible points

    # Combine all points and colors
    all_points = np.vstack([obstacle_vertices, feasible_points, infeasible_points])
    all_colors = np.vstack([obstacle_colors, feasible_colors, infeasible_colors])

    # Create a point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points)
    point_cloud.colors = o3d.utility.Vector3dVector(all_colors / 255.0)  # Normalize colors to [0, 1]

    # Save the result as a PLY file
    o3d.io.write_point_cloud(ply_file, point_cloud)
    print(f"Saved the result to {ply_file}")

    
def get_world_config(objs: List[Entity], return_center=False) -> WorldConfig:
    obstacles = {"mesh": []}
    print("get world config start")
    pattern = r'\((\d+)\)[^()]*\((\d+)\)'

    ignore_asset_id = []
    id_to_placeholder_idx = {}

    room_center = np.zeros(3)
    for i, obj in enumerate(objs):
        if isinstance(obj, MeshObject):
            scale = obj.get_scale()
            name = obj.get_name()
            if "living-room" in name.lower() and "floor" in name.lower():
                room_center = obj.get_location()

            if  "ceiling" in name.lower() or "exterior" in name.lower() or "floor" in name.lower()\
                    or "rug" in name.lower() or "door" in name.lower() or "window" in name.lower() or \
                    "open" in name.lower() or "hoof" in name.lower():
                continue
            if "0/0" in name.lower() and "wall" not in name.lower():
                continue
                
            if "placeholder" in name.lower():
                match = re.search(pattern, name)
                id = match.group(2)
                id_to_placeholder_idx[id] = i
                continue

            blender_mesh = obj.get_mesh()

            vertices = np.array([vertex.co[:] for vertex in blender_mesh.vertices])
            if vertices.shape[0] > 4.0e4:
                match = re.search(pattern, name)
                if match is None:
                    continue
                id = match.group(2)
                ignore_asset_id.append(id)
                continue
                
            matrix_world = np.array(obj.get_local2world_mat())
            tensor_mat = tensor_args.to_device(matrix_world)
            pose = Pose.from_matrix(tensor_mat).tolist()
                    
            faces = []
            bad_mesh = False
            for i, polygon in enumerate(blender_mesh.polygons):
                face_vertices = vertices[np.array(polygon.vertices[:3])] 
                face_vertices[:, 2] += matrix_world[2, 3]
                if any(face_vertices[:, 2] > 5) or any(face_vertices[:, 2] < -5):
                    bad_mesh = True
                on_ground = face_vertices[:, 2] < 0.2
                if np.count_nonzero(on_ground) == 3:
                    continue
                faces.append(np.array(polygon.vertices[:3]))
            if bad_mesh:
                continue
            if len(faces) == 0:
                continue
            
            print(f"name {name}, vertices num: {vertices.shape}  faces num: {len(faces)}")


            curobo_mesh = Mesh(
                name=name,
                pose=pose,
                vertices=vertices.tolist(),
                faces=faces,
                scale=scale
            )

            obstacles["mesh"].append(curobo_mesh)

    for id in ignore_asset_id:
        if id in id_to_placeholder_idx:
            obj = objs[id_to_placeholder_idx[id]]
            scale = obj.get_scale()
            name = obj.get_name()
            print(f"add placehold for {name}     id {id}")

            blender_mesh = obj.get_mesh()

            vertices = np.array([vertex.co[:] for vertex in blender_mesh.vertices])
            
            faces = np.zeros((len(blender_mesh.polygons), 3))
            for i, polygon in enumerate(blender_mesh.polygons):
                faces[i] = polygon.vertices[:3]
        
            matrix_world = np.array(obj.get_local2world_mat())
            tensor_mat = tensor_args.to_device(matrix_world)
            pose = Pose.from_matrix(tensor_mat).tolist()

            curobo_mesh = Mesh(
                name=name,
                pose=pose,
                vertices=vertices.tolist(),
                faces=faces.tolist(),
                scale=scale
            )

            obstacles["mesh"].append(curobo_mesh)
                
    world_model = WorldConfig(**obstacles)

    if return_center:
        return world_model, room_center
    return world_model

def get_start_and_goal(objs : list):
    start_poses = []
    end_poses = []
    for obj in objs:
        if isinstance (obj, MeshObject):
            # import ipdb; ipdb.set_trace()
            if "living-room" in obj.get_name() and "floor" in obj.get_name():
                start_poses.append(obj.get_origin())
                print(f"hit {obj.get_name()}, location is {obj.get_origin()}" )
    
    return start_poses, end_poses

def cs_to_bs(curobo_state : np.ndarray):
    # blender state是 x y z yaw j1 j2 j3 j4 j5 j6 j7
    # curobo state是 x y yaw j1 j2 j3 j4 j5 j6 j7 0
    # curobo state: [batch, traj_len, dof]
    B, T, D = curobo_state.shape
    blender_state = np.zeros((B, T, 12))
    blender_state[:, :, :2] = curobo_state[:, :, :2]
    blender_state[:, :, 2] = 0.2
    blender_state[:, :, 3] = curobo_state[:, :, 2]
    blender_state[:, :, 4:11] = curobo_state[:, :, 3:]
    
    return blender_state    

def set_robots_state(robots :  List[URDFObject], state : np.ndarray):
    BASE_SHIFT = 4
    if len(state.shape) == 1:
        state = np.tile(state, (NUM_ROBOTS, 1))

    for i, robot in enumerate(robots):
        robot.set_location(state[i][:3]) #x y z
        robot.set_rotation_euler([0, 0, state[i][3]]) #yaw
        curobo_state[i * ROBOT_SEED : (i + 1) * ROBOT_SEED, :2] = state[i][:2] # x y
        curobo_state[i * ROBOT_SEED : (i + 1) * ROBOT_SEED, 2] = state[i][3] # yaw
        for j, link in enumerate(robot.get_links_with_revolute_joints()):
            robot.set_rotation_euler_fk(link, rotation_euler=state[i][j+BASE_SHIFT], mode='absolute')
            curobo_state[i * ROBOT_SEED : (i + 1) * ROBOT_SEED, j+3] = state[i][j+BASE_SHIFT] # joint angles

def get_init_poses(objs: list, room_center: np.ndarray):
    init_arm_pose = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0])
    MAX_ATTEMPS = 50

    check_state = np.zeros(10)
    check_state[:2] = room_center[:2]
    check_state[2] = 0.0  # yaw
    check_state[3:] = init_arm_pose
    check_state_curobo = JointState.from_position(tensor_args.to_device(check_state), joint_names=motion_gen.rollout_fn.joint_names)
    valid, _ = motion_gen.check_start_state(check_state_curobo)
    sph_list = motion_gen.kinematics.get_robot_as_spheres(tensor_args.to_device(check_state))
    for si, s in enumerate(sph_list[0]):
        bproc.object.create_primitive("SPHERE", scale=[s.radius, s.radius, s.radius], location=s.position)
    if valid:
        return check_state

    for i in range(MAX_ATTEMPS):
        max_range = np.clip(0.5*i, 0, 10)
        x = np.random.uniform(-max_range, max_range)
        y = np.random.uniform(-max_range, max_range)
        # yaw = np.random.uniform(-np.pi / 2, np.pi / 2)
        check_state[:2] = room_center[:2] + np.array([x, y])
        if check_state[0] < 1.0  or check_state[1] < 1.0 :
            continue
        check_state[2] = 0
        check_state_curobo = JointState.from_position(tensor_args.to_device(check_state), joint_names=motion_gen.rollout_fn.joint_names)
        valid, _ = motion_gen.check_start_state(check_state_curobo)
        

            
        if valid:
            return check_state

    raise ValueError("can not find feasible initial pose")

# region Callback
def motion_plan_callback():
    global last_success_target, cmd_idx, cmd_trajs, task_finish, cube_pos_history, cmd_step_size
    print(f"Motion plan is running... Cube position: {cube.get_location()}. task_finish: {task_finish} \
        cmd_idx{cmd_idx}/{cmd_trajs.shape[1] if cmd_trajs is not None else 0}")
    
    #######################################motion control################################################
    if cmd_trajs is not None and not task_finish:
        set_robots_state(robots, cmd_trajs[:, cmd_idx])
        cmd_idx += cmd_step_size
        if cmd_idx >= cmd_trajs.shape[1]:
            task_finish = True
            cmd_idx = 0
            cmd_trajs = None
        return 0.15
    #####################################################################################################
    cube_position = cube.get_location()
    cube_position[2] = np.clip(cube_position[2], 0.4, 1.0)
    if np.linalg.norm(cube_position - last_success_target) > 0.2 and \
        np.linalg.norm(cube_pos_history[0] - cube_position) == 0.0:
        ik_goal = Pose(
            position=tensor_args.to_device(cube_position),
            quaternion=tensor_args.to_device([0, 1, 0, 0]),
        )
        curobo_joint_state = JointState.from_position(tensor_args.to_device(curobo_state[:, :10]), joint_names=motion_gen.rollout_fn.joint_names)

        result = motion_gen.plan_batch(
                        curobo_joint_state.clone(),
                        ik_goal.clone().repeat_seeds(NUM_ROBOTS * ROBOT_SEED),
                        plan_config,
                )
        if result.optimized_plan is None:
            print(f"result.success {result.success}   result.status: {result.status}")
            return 0.05
        all_trajs = motion_gen.get_full_js(result.optimized_plan).position.cpu().numpy()[:, :, :10]
        plan_success = np.zeros(NUM_ROBOTS, dtype=bool)
        suceess_trajs = []
        for i in range(NUM_ROBOTS * ROBOT_SEED):
            if result.success[i] and not plan_success[i // ROBOT_SEED]:
                plan_success[i // ROBOT_SEED] = True
                suceess_trajs.append(all_trajs[i])
        
        # if all robots has success plan        
        if np.count_nonzero(plan_success) == NUM_ROBOTS: 
            cmd_trajs = cs_to_bs(np.array(suceess_trajs))
            cmd_step_size = cmd_trajs.shape[1] // 32
            cmd_idx = 0
            task_finish = False
            last_success_target = cube_position
        else:
            print(f"result.success {result.success}   result.status: {result.status}")
            
        print(f"result ik time:{result.ik_time:.3f} graph time:{result.graph_time:.3f} " \
                f"opt_time:{result.trajopt_time:.3f} finetune:{result.finetune_time:.3f} total:{result.total_time:.3f} " \
                f"attemps:{result.attempts} opt_attemps:{result.trajopt_attempts}")
        
        if not result.success[0]:
            return 0.05
    cube_pos_history[0] = cube_pos_history[1]
    cube_pos_history[1] = cube_position
    return 1.0  


# region Main
bproc.init()
cube = bproc.object.create_primitive("CUBE", scale=[0.05, 0.05, 0.05], location=[0, 0, 0])
objs = bproc.loader.load_blend(
    path="/ssd/yangyuqiang/infinigen/outputs/multi_dataset_big_door/784188a8/fine/scene.blend",
    # path="/ssd/yangyuqiang/infinigen/outputs/multi_dataset_no_plantandshlefobj/14ec7b18/fine/scene.blend",
    # path="/ssd/yangyuqiang/infinigen/outputs/multi_dataset_big_door/77f33467/fine/scene.blend",
    obj_types=['mesh', 'curve', 'hair', 'armature','empty', 'light', 'camera'],
    data_blocks=['armatures', 'cameras', 'collections', 'curves', 'images', 'lights', 'materials', 'meshes', 'objects', 'textures'])
hide_collection("unique_assets:room_exterior")
hide_collection("unique_assets:room_ceiling")
bpy.context.window.workspace = bpy.data.workspaces["yuqiang"]
bpy.context.view_layer.update()

# load robot

robots = []
tensor_args = TensorDeviceType()
urdf_file="/ssd/yangyuqiang/curobo/src/curobo/content/assets/robot/ridgeback_franka/RidgebackFranka.urdf"
curobo_world_config, room_center = get_world_config(objs, True)
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = [executor.submit(get_world_config, objs, True)]
    
for i in range(NUM_ROBOTS):
    robot = bproc.loader.load_urdf(urdf_file="/ssd/yangyuqiang/curobo/src/curobo/content/assets/robot/ridgeback_franka/RidgebackFranka.urdf" + str(i))
    print("load success")
    robots.append(robot)

    # for future in concurrent.futures.as_completed(futures):
    #     curobo_world_config, room_center = future.result()
    
hide_inertial()



# region Curobo
setup_curobo_logger("warn")
n_obstacle_mesh = 2000
n_obstacle_cuboids = 300
robot_cfg_path = get_robot_configs_path()
robot_cfg = load_yaml(join_path(robot_cfg_path, "ridgeback_franka.yml"))["robot_cfg"]

j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
world_cfg = WorldConfig()

# curobo parameter
trajopt_dt = None
optimize_dt = False
trajopt_tsteps = 32
trim_steps = None
max_attempts = 2
interpolation_dt = 0.05
enable_finetune_trajopt = True             

motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_cfg,
    world_cfg,
    tensor_args,
    collision_checker_type=CollisionCheckerType.MESH,
    num_trajopt_seeds=64,
    num_graph_seeds=12,
    interpolation_dt=interpolation_dt,
    collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
    optimize_dt=optimize_dt,
    trajopt_dt=trajopt_dt,
    trajopt_tsteps=trajopt_tsteps,
    trim_steps=trim_steps,
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, batch=NUM_ROBOTS * ROBOT_SEED) 
print(f"curobot is ready")


plan_config = MotionGenPlanConfig(
    enable_graph=True,
    enable_graph_attempt=2,
    max_attempts=max_attempts,
    enable_finetune_trajopt=enable_finetune_trajopt,
)

motion_gen.update_world(curobo_world_config.clone())

start_pose = get_init_poses(objs, room_center)

start_pose = np.insert(start_pose, 2, 0.2)  # fake z position
start_pose = np.insert(start_pose, -1, 0.0)
set_robots_state(robots, start_pose)

# add the target cube
curobo_joint_state = JointState.from_position(tensor_args.to_device(curobo_state),joint_names=motion_gen.rollout_fn.joint_names)
goal_state = motion_gen.rollout_fn.compute_kinematics(curobo_joint_state)
ee_pose = Pose(goal_state.ee_pos_seq, quaternion=goal_state.ee_quat_seq)
cube.set_location(ee_pose.position[0].cpu().numpy())

collision_supported_world = WorldConfig.create_collision_support_world(curobo_world_config)
collision_supported_world.save_world_as_mesh("debug_collision_mesh.obj")

# collision_map = get_collision_map(objs, room_center, motion_gen, tensor_args)
# obj_to_ply_with_collision_map("debug_collision_mesh.obj", "debug2.ply", collision_map)

# region Timerg
timer2 = bpy.app.timers.register(motion_plan_callback)

# while True:
#     bpy.context.view_layer.update()


print("finish")