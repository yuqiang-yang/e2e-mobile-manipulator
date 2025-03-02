import blenderproc as bproc
import concurrent.futures
import numpy as np
import bpy
import torch
import argparse
import os
import re
import cv2
from pathlib import Path
from typing import List
from scipy.spatial.transform import Rotation as R

from blenderproc.python.types.MeshObjectUtility import MeshObject, create_primitive
from blenderproc.python.types.URDFUtility import URDFObject
from blenderproc.python.types.EntityUtility import Entity, convert_to_entity_subclass

# curobo
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig, CollisionQueryBuffer
from curobo.geom.sdf.world_mesh import WorldMeshCollision
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
    MotionGenResult
)

# region Global Variables
NUM_ROBOTS = 2
ROBOT_SEED = 4
last_success_target = np.zeros(3)
success_result = None
curobo_state = np.zeros((NUM_ROBOTS * ROBOT_SEED, 12))
task_finish = True
cmd_idx = 0
cmd_trajs = None
cmd_step_size = 1
cube_position = np.zeros(3)
failure_cnt = 0
plan_id = 0  # Plan ID counter

# endregion

# region Helper Functions


def hide_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection:
        collection.hide_viewport = True
        collection.hide_render = True
        collection.hide_select = True


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

            if "ceiling" in name.lower() or "exterior" in name.lower() or "floor" in name.lower() \
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

    for id in ignore_asset_id:
        if id in id_to_placeholder_idx:
            obj = objs[id_to_placeholder_idx[id]]
            scale = obj.get_scale()
            name = obj.get_name()

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


def get_targets(objs: list, world_ccheck: WorldMeshCollision):
    target_pose = []
    added_id = []
    pattern = r'\((\d+)\)[^()]*\((\d+)\)'

    for obj in objs:
        if isinstance(obj, MeshObject):
            name = obj.get_name()
            match = re.search(pattern, name)
            if match is not None:
                id = match.group(2)
                if id in added_id:
                    continue

            bbox = obj.get_bound_box()
            w_T_o = obj.get_local2world_mat()

            bbox_in_local = np.linalg.inv(w_T_o) @ np.concatenate([bbox, np.ones((8, 1))], axis=1).T
            assert bbox_in_local.shape[1] == 8
            length, width, height, _ = np.max(bbox_in_local, axis=1) - np.min(bbox_in_local, axis=1)

            min_z = np.min(bbox[:, 2])
            max_z = np.max(bbox[:, 2])

            if min_z > 0.15:
                continue

            if max_z < 0.3 or max_z > 2.6:
                continue

            origin_at_corner = False
            if np.any(np.linalg.norm(bbox_in_local[:3], axis=0) < 6e-2):
                origin_at_corner = True

            OFFSET = 0.1
            if max_z > 1.4:
                candidate_pose = (w_T_o @ np.array([length + OFFSET, width / 2 if origin_at_corner else 0, height / 2, 1]))[:3]
            else:
                candidate_pose = (w_T_o @ np.array([length, width / 2 if origin_at_corner else 0, height + OFFSET, 1]))[:3]

            radius = 0.1
            candidate_radius = np.concatenate([candidate_pose, [radius]])
            candidate_radius = tensor_args.to_device(candidate_radius).view(1, 1, 1, 4)
            query_buffer = CollisionQueryBuffer.initialize_from_shape(
                candidate_radius.shape, tensor_args, world_ccheck.collision_types
            )
            act_distance = tensor_args.to_device([0.0])
            weight = tensor_args.to_device([1])
            collide = world_ccheck.get_sphere_collision(candidate_radius, query_buffer, weight, act_distance)
            collide = collide.view(1)

            if collide:
                continue

            candidate_pose[2] = np.clip(candidate_pose[2], 0, 1.5)
            target_pose.append(candidate_pose)

            match = re.search(pattern, name)
            if match is not None:
                id = match.group(2)
                added_id.append(id)
            
    return np.array(target_pose)


def get_init_poses(objs: list, room_center: np.ndarray):
    init_arm_pose = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0])
    MAX_ATTEMPS = 50

    check_state = np.zeros(10)
    check_state[:2] = room_center[:2]
    check_state[2] = 0.0  # yaw
    check_state[3:] = init_arm_pose
    check_state_curobo = JointState.from_position(tensor_args.to_device(check_state), joint_names=motion_gen.rollout_fn.joint_names)
    valid, _ = motion_gen.check_start_state(check_state_curobo)
    if valid:
        return check_state

    for i in range(MAX_ATTEMPS):
        max_range = np.clip(0.5*i, 0, 10)
        x = np.random.uniform(-max_range, max_range)
        y = np.random.uniform(-max_range, max_range)
        # yaw = np.random.uniform(-np.pi / 2, np.pi / 2)
        check_state[:2] = room_center[:2] + np.array([x, y])
        check_state[2] = 0
        check_state_curobo = JointState.from_position(tensor_args.to_device(check_state), joint_names=motion_gen.rollout_fn.joint_names)
        valid, _ = motion_gen.check_start_state(check_state_curobo)

        if valid:
            return check_state

    raise ValueError("can not find feasible initial pose")


def select_random_goal(candidates: np.ndarray, threshold=5.0):
    for i in range(candidates.shape[0]):
        idx = np.random.randint(0, candidates.shape[0] - 1)
        if np.linalg.norm(candidates[idx] - cube_position) > threshold:
            return np.copy(candidates[idx])
    raise ValueError("select_random_goal can select feasible goal")


def cs_to_bs(curobo_state: np.ndarray):
    B, T, D = curobo_state.shape
    blender_state = np.zeros((B, T, 12))
    blender_state[:, :, :2] = curobo_state[:, :, :2]
    blender_state[:, :, 2] = 0.2
    blender_state[:, :, 3] = curobo_state[:, :, 2]
    blender_state[:, :, 4:11] = curobo_state[:, :, 3:]
    return blender_state


def set_robots_state(robots: List[URDFObject], state: np.ndarray):
    BASE_SHIFT = 4
    if len(state.shape) == 1:
        state = np.tile(state, (NUM_ROBOTS, 1))

    for i, robot in enumerate(robots):
        robot.set_location(state[i][:3])  # x y z
        robot.set_rotation_euler([0, 0, state[i][3]])  # yaw
        curobo_state[i * ROBOT_SEED: (i + 1) * ROBOT_SEED, :2] = state[i][:2]  # x y
        curobo_state[i * ROBOT_SEED: (i + 1) * ROBOT_SEED, 2] = state[i][3]  # yaw
        for j, link in enumerate(robot.get_links_with_revolute_joints()):
            robot.set_rotation_euler_fk(link, rotation_euler=state[i][j + BASE_SHIFT], mode='absolute')
            curobo_state[i * ROBOT_SEED: (i + 1) * ROBOT_SEED, j + 3] = state[i][j + BASE_SHIFT]  # joint angles


# endregion

# region Render Function
def render_and_save_images(robots, cmd_trajs, plan_id):
    def create_camera_pose_from_euler(pitch, yaw, roll, position):
        rotation = R.from_euler('xyz', [pitch, roll, yaw], degrees=False).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        return pose
    def transform_point(transform, point):
        point_homogeneous = np.append(point, 1)  
        transformed_point = transform @ point_homogeneous
        return transformed_point[:3]
    for robot_id , robot in enumerate(robots):
        robot_output_dir = os.path.join(output_dir, f"{plan_id}", f"robot_{robot_id}")
        hide_other_robots(robots, robot_id)
        for frame_idx in range(cmd_trajs.shape[1]):
            set_robot_pose_at_frame(robot, cmd_trajs[robot_id, frame_idx], 2 * frame_idx)
            set_robot_pose_at_frame(robot, cmd_trajs[robot_id, frame_idx], 2 * frame_idx + 1)
            yaw = cmd_trajs[robot_id, frame_idx, 3]

            first_person_position = [0.5, 0, 0.3]
            robot_pose = create_camera_pose_from_euler(0, yaw, 0, cmd_trajs[robot_id, frame_idx][:3])
            # first_person_pose = create_camera_pose_from_euler(np.pi / 2, yaw, 0, transform_point(robot_pose, first_person_position))
            first_person_pose = bproc.math.build_transformation_mat(transform_point(robot_pose, first_person_position), [np.pi / 2, 0, yaw - np.pi/2])
            third_person_position = [-0.7, 0, 1.5]
            # third_person_pose = create_camera_pose_from_euler(np.pi / 3, yaw, 0, transform_point(robot_pose, third_person_position))
            third_person_pose = bproc.math.build_transformation_mat(transform_point(robot_pose, third_person_position), [np.pi / 3, 0, yaw - np.pi/2])
            bproc.camera.add_camera_pose(first_person_pose, frame=2 * frame_idx)
            bproc.camera.add_camera_pose(third_person_pose, frame=2 * frame_idx + 1)

            
        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end
        data = bproc.renderer.render()

        os.makedirs(os.path.join(robot_output_dir, "rgb", "first"), exist_ok=True)
        os.makedirs(os.path.join(robot_output_dir, "depth", "first"), exist_ok=True)
        os.makedirs(os.path.join(robot_output_dir, "rgb", "third"), exist_ok=True)
        os.makedirs(os.path.join(robot_output_dir, "depth", "third"), exist_ok=True)
        first_rgb_writer = cv2.VideoWriter(os.path.join(robot_output_dir, "first_rgb.mp4"), 
                        cv2.VideoWriter_fourcc(*'mp4v'), 10, (args.image_width, args.image_height))
        first_depth_writer = cv2.VideoWriter(os.path.join(robot_output_dir, "first_depth.mp4"), 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 10, (args.image_width, args.image_height))
        third_rgb_writer = cv2.VideoWriter(os.path.join(robot_output_dir, "third_rgb.mp4"), 
                        cv2.VideoWriter_fourcc(*'mp4v'), 10, (args.image_width, args.image_height))
        third_depth_writer = cv2.VideoWriter(os.path.join(robot_output_dir, "third_depth.mp4"), 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 10, (args.image_width, args.image_height))
        for frame_idx in range(cmd_trajs.shape[1]):
            rgb_path_first = os.path.join(robot_output_dir, "rgb", "first" ,f"rgb{frame_idx:03d}.jpg")
            depth_path_first = os.path.join(robot_output_dir, "depth", "first", f"depth{frame_idx:03d}.png")
            depth = data["depth"][frame_idx * 2] 
            rgb = cv2.cvtColor(data["colors"][frame_idx * 2], cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_path_first, rgb)
            cv2.imwrite(depth_path_first, depth * 1000)
            depth_normalized = (np.clip(depth / 10.0, 0, 1) * 255.0).astype(np.uint8)
            first_rgb_writer.write(rgb)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            first_depth_writer.write(depth_colored)

            rgb_path_third = os.path.join(robot_output_dir, "rgb", "third" ,f"rgb{frame_idx:03d}.jpg")
            depth_path_third = os.path.join(robot_output_dir, "depth", "third" ,f"depth{frame_idx:03d}.png")
            depth = data["depth"][frame_idx * 2 + 1] 
            rgb = cv2.cvtColor(data["colors"][frame_idx * 2 + 1], cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_path_third, rgb)
            depth_normalized = (np.clip(depth / 10.0, 0, 1) * 255.0).astype(np.uint8)
            cv2.imwrite(depth_path_third, depth * 1000)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            third_rgb_writer.write(rgb)
            third_depth_writer.write(depth_colored)
        
        first_rgb_writer.release()
        first_depth_writer.release()
        third_rgb_writer.release()
        third_depth_writer.release()

# endregion

def hide_other_robots(robots: List, current_robot_id: int):
    for i, robot in enumerate(robots):
        robot.hide(i != current_robot_id)

def set_robot_pose_at_frame(robot: bproc.types.URDFObject, traj, frame_idx: int):
    robot.set_location(traj[:3], frame=frame_idx)  # x, y, z
    robot.set_rotation_euler([0, 0, traj[3]], frame=frame_idx)  # yaw
    for j, link in enumerate(robot.get_links_with_revolute_joints()):
        robot.set_rotation_euler_fk(link, rotation_euler=traj[j + 4], mode='absolute', frame=frame_idx)

# region Save Trajectory


def save_trajectory(cmd_trajs, plan_id):
    for robot_id in range(NUM_ROBOTS):
        robot_output_dir = os.path.join(output_dir, f"{plan_id}", f"robot_{robot_id}")
        os.makedirs(robot_output_dir, exist_ok=True)

        traj_path = os.path.join(robot_output_dir, "trajectory.npy")
        np.save(traj_path, cmd_trajs[robot_id])


# endregion

# region Motion Plan Callback


def motion_plan_callback():
    global last_success_target, cmd_idx, cmd_trajs, task_finish, cmd_step_size, cube_position, failure_cnt, plan_id

    print(f"Motion plan is running... Cube position: {cube.get_location()}. task_finish: {task_finish} "
          f"cmd_idx {cmd_idx}/{cmd_trajs.shape[1] if cmd_trajs is not None else 0}")

    if failure_cnt > 1:
        cube_position = select_random_goal(target_poses)
        cube.set_location(cube_position)
        failure_cnt = 0

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

    ik_goal = Pose(
        position=tensor_args.to_device(cube_position),
        quaternion=tensor_args.to_device([0, 1, 0, 0]),
    )
    curobo_joint_state = JointState.from_position(tensor_args.to_device(curobo_state[:, :10]),
                                                  joint_names=motion_gen.rollout_fn.joint_names)

    result = motion_gen.plan_batch(
        curobo_joint_state.clone(),
        ik_goal.clone().repeat_seeds(NUM_ROBOTS * ROBOT_SEED),
        plan_config,
    )
    if result.optimized_plan is None:
        print(f"result.success {result.success}   result.status: {result.status}")
        failure_cnt += 1
        return 0.05

    all_trajs = motion_gen.get_full_js(result.optimized_plan).position.cpu().numpy()[:, :, :10]
    plan_success = np.zeros(NUM_ROBOTS, dtype=bool)
    success_trajs = []
    for i in range(NUM_ROBOTS * ROBOT_SEED):
        if result.success[i] and not plan_success[i // ROBOT_SEED]:
            plan_success[i // ROBOT_SEED] = True
            success_trajs.append(all_trajs[i])

    if np.count_nonzero(plan_success) == NUM_ROBOTS:
        cmd_trajs = cs_to_bs(np.array(success_trajs))
        cmd_step_size = cmd_trajs.shape[1] // 32
        cmd_idx = 0
        task_finish = False
        last_success_target = cube_position

        save_trajectory(cmd_trajs, plan_id)
        print(f"start state {curobo_joint_state.position[0]}")
        render_and_save_images(robots, cmd_trajs, plan_id)
        cube_position = select_random_goal(target_poses)
        cube.set_location(cube_position)
        failure_cnt = 0

        plan_id += 1
    else:
        print(f"result.success {result.success}   result.status: {result.status}")
        failure_cnt += 1

    print(f"result ik time:{result.ik_time:.3f} graph time:{result.graph_time:.3f} "
          f"opt_time:{result.trajopt_time:.3f} finetune:{result.finetune_time:.3f} total:{result.total_time:.3f}  "
          f"attemps:{result.attempts} opt_attemps:{result.trajopt_attempts}")

    if not result.success[0]:
        return 0.05

    return 1.0


# endregion


def setup_studio_light(studio_light_name, strength=1.0):
    blender_data_path = bpy.utils.resource_path('LOCAL')
    studio_light_dir = os.path.join(blender_data_path, "datafiles", "studiolights", "world")
    
    studio_light_path = os.path.join(studio_light_dir, f"{studio_light_name}")
    
    if not os.path.exists(studio_light_path):
        raise FileNotFoundError(f"Studio Light 文件未找到: {studio_light_path}")
    
    world = bpy.context.scene.world
    world.use_nodes = True
    
    nodes = world.node_tree.nodes
    nodes.clear()
    
    env_texture = nodes.new(type="ShaderNodeTexEnvironment")
    env_texture.image = bpy.data.images.load(studio_light_path)
    
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Strength"].default_value = strength  # 设置环境光强度
    
    output = nodes.new(type="ShaderNodeOutputWorld")
    
    links = world.node_tree.links
    links.new(env_texture.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])

    print(f"已成功加载 Studio Light: {studio_light_name}")
    
def generate_intrinsic(width,height,hfov,vfov):
    intrinsic = np.eye(3)
    intrinsic[0][0] = width / (2 * (np.tan(np.deg2rad(hfov)/2)))
    intrinsic[1][1] = height / (2 * (np.tan(np.deg2rad(vfov)/2)))
    intrinsic[0][2] = width / 2
    intrinsic[1][2] = height / 2
    return intrinsic

# region Main


parser = argparse.ArgumentParser()
parser.add_argument("--scene_path", type=str, default="/ssd/yangyuqiang/infinigen/outputs/multi_dataset_big_door/77f33467/fine/scene.blend")
#/ssd/yangyuqiang/infinigen/outputs/multi_dataset_big_door_less_obs/2a14347
parser.add_argument("--trajs", type=int, default=10)
parser.add_argument("--image_height",type=int,default=480)
parser.add_argument("--image_width",type=int,default=640)
parser.add_argument("--camera_hfov",type=float,default=86)
parser.add_argument("--camera_vfov",type=float,default=57)
parser.add_argument("--device",type=int,default=3)

args = parser.parse_args()
# Extract scene_id from the input scene_path
scene_id = Path(args.scene_path).parts[-3]
output_dir = os.path.join("output", scene_id)
os.makedirs(output_dir, exist_ok=True)


bproc.init()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
# bproc.renderer.set_render_devices(False,"CUDA",args.device)
bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
camera_intrinsic = generate_intrinsic(args.image_width,args.image_height,args.camera_hfov,args.camera_vfov)
bproc.camera.set_intrinsics_from_K_matrix(camera_intrinsic,args.image_width,args.image_height)
setup_studio_light("forest.exr", strength=1.0)

cube = bproc.object.create_primitive("CUBE", scale=[0.01, 0.01, 0.01], location=[0, 0, 0])
objs = bproc.loader.load_blend(
    args.scene_path,
    obj_types=['mesh', 'curve', 'hair', 'armature', 'empty', 'light', 'camera'],
    data_blocks=['armatures', 'cameras', 'collections', 'curves', 'images', 'lights', 'materials', 'meshes', 'objects', 'textures']
)
# hide_collection("unique_assets:room_exterior")
# hide_collection("unique_assets:room_ceiling")
bpy.context.window.workspace = bpy.data.workspaces["yuqiang"]
bpy.context.view_layer.update()

robots = []
tensor_args = TensorDeviceType()
urdf_file = "/ssd/yangyuqiang/curobo/src/curobo/content/assets/robot/ridgeback_franka/RidgebackFranka.urdf"

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(get_world_config, objs, True)]

    for i in range(NUM_ROBOTS):
        robot = bproc.loader.load_urdf(urdf_file=urdf_file + str(i))
        robot.remove_link_by_index(index=0)

        print("load success")
        robots.append(robot)

    for future in concurrent.futures.as_completed(futures):
        curobo_world_config, room_center = future.result()
setup_curobo_logger("warn")
n_obstacle_mesh = 600
n_obstacle_cuboids = 50
robot_cfg_path = get_robot_configs_path()
robot_cfg = load_yaml(join_path(robot_cfg_path, "ridgeback_franka.yml"))["robot_cfg"]

j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
world_cfg = WorldConfig()

trajopt_dt = None
optimize_dt = False
trajopt_tsteps = 32
trim_steps = None
max_attempts = 2
interpolation_dt = 0.05
enable_finetune_trajopt = False

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

collision_supported_world = WorldConfig.create_collision_support_world(curobo_world_config)
# collision_supported_world.save_world_as_mesh("debug_collision_mesh.obj")

world_collision_config = WorldCollisionConfig(tensor_args, world_model=collision_supported_world)
world_ccheck = WorldMeshCollision(world_collision_config)

motion_gen.update_world(curobo_world_config.clone())

target_poses = get_targets(objs, world_ccheck)
start_pose = get_init_poses(objs, room_center)

start_pose = np.insert(start_pose, 2, 0.2)  # fake z position
start_pose = np.insert(start_pose, -1, 0.0)
set_robots_state(robots, start_pose)
curobo_joint_state = JointState.from_position(tensor_args.to_device(curobo_state), joint_names=motion_gen.rollout_fn.joint_names)
goal_state = motion_gen.rollout_fn.compute_kinematics(curobo_joint_state)
ee_pose = Pose(goal_state.ee_pos_seq, quaternion=goal_state.ee_quat_seq)
cube.set_location(ee_pose.position[0].cpu().numpy())
cube_position = ee_pose.position[0].cpu().numpy()
for _ in range(1000):
    motion_plan_callback()
# timer2 = bpy.app.timers.register(motion_plan_callback)

# endregion