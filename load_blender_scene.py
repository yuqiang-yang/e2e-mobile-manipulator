import blenderproc as bproc
import concurrent.futures
import numpy as np
import bpy
import torch
import argparse
import time
import re

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
    MotionGenResult
)

# region Global
#####################Global Variables#######################
NUM_ROBOTS = 2
last_success_target = np.zeros(3)
success_result = None
curobo_state = np.zeros((NUM_ROBOTS, 12)) 
task_finish = True
cmd_idx = 0
cmd_trajs = None
############################################################

# region Function
def hide_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection:
        collection.hide_viewport = True
        collection.hide_render = True
        collection.hide_select = True

def get_world_config(objs : List[Entity]) -> WorldConfig:
    obstacles = {"mesh" : []}
    print("get world config start")
    pattern = r'\((\d+)\)[^()]*\((\d+)\)'

    ignore_asset_id = []
    id_to_placeholder_idx = {}
    for i, obj in enumerate(objs):
        if isinstance(obj, MeshObject):
            scale = obj.get_scale()
            name = obj.get_name()
            if "ceiling" in name.lower() or "exterior" in name.lower() or "floor" in name.lower() \
                or "rug" in name.lower() or "door" in name.lower() or "window" in name.lower() or \
                    "open" in name.lower():
                continue
            if "0/0" in name.lower() and "wall" not in name.lower():
                continue

            if "placeholder" in name.lower():
                match = re.search(pattern, name)
                id = match.group(2)
                id_to_placeholder_idx[id] = i
                print(f"########### {name}", " id", id)
                continue
            
            blender_mesh = obj.get_mesh()

            vertices = np.array([vertex.co[:] for vertex in blender_mesh.vertices])
            if vertices.shape[0] > 4.0e4:
                match = re.search(pattern, name)
                id = match.group(2)
                ignore_asset_id.append(id)
                print(f"********* {name}", " id", id)
                continue
            # faces = []
            # for polygon in blender_mesh.polygons:
            #     face_vertices = polygon.vertices[:]
            #     if len(face_vertices) == 3:
            #         faces.append(face_vertices)
            #     elif len(face_vertices) > 3:
            #         for i in range(1, len(face_vertices) - 1):
            #             faces.append([face_vertices[0], face_vertices[i], face_vertices[i + 1]])
            
            faces = np.zeros((len(blender_mesh.polygons), 3))
            for i, polygon in enumerate(blender_mesh.polygons):
                faces[i] = polygon.vertices[:3]
        
            matrix_world = np.array(obj.get_local2world_mat())
            tensor_mat = tensor_args.to_device(matrix_world)
            pose = Pose.from_matrix(tensor_mat).tolist()


            # print(f"name {name}, vertices num: {vertices.shape}  faces num: {len(faces)}")

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
        curobo_state[i][:2] = state[i][:2] # x y
        curobo_state[i][2] = state[i][3] # yaw
        for j, link in enumerate(robot.get_links_with_revolute_joints()):
            robot.set_rotation_euler_fk(link, rotation_euler=state[i][j+BASE_SHIFT], mode='absolute')
            curobo_state[i][j+3] = state[i][j+BASE_SHIFT] # joint angles
            
# region Callback
def motion_plan_callback():
    global last_success_target, cmd_idx, cmd_trajs, task_finish
    print(f"Motion plan is running... Cube position: {cube.get_location()}. task_finish: {task_finish}")
    
    #######################################motion control################################################
    if cmd_trajs is not None and not task_finish:
        set_robots_state(robots, cmd_trajs[:, cmd_idx])
        cmd_idx += 1
        if cmd_idx >= cmd_trajs.shape[1]:
            task_finish = True
            cmd_idx = 0
            cmd_trajs = None
        return 0.15
    #####################################################################################################
    cube_position = cube.get_location()
    cube_position[2] = np.clip(cube_position[2], 0.4, 1.0)
    if np.linalg.norm(cube_position - last_success_target) > 0.2:
        ik_goal = Pose(
            position=tensor_args.to_device(cube_position),
            quaternion=tensor_args.to_device([0, 1, 0, 0]),
        )
        print(f"start motion plan")
        curobo_joint_state = JointState.from_position(tensor_args.to_device(curobo_state[:, :10]), joint_names=motion_gen.rollout_fn.joint_names)
        tt = time.time()
        # import ipdb; ipdb.set_trace()
        try:
            result = motion_gen.plan_batch(
                            curobo_joint_state.clone(),
                            ik_goal.clone().repeat_seeds(NUM_ROBOTS),
                            plan_config,
                    )
        except Exception as e:
            print("plan inner error",e)
            return 0.05
            # import ipdb; ipdb.set_trace()
        if result.success[0]:
            cmd_trajs = cs_to_bs(motion_gen.get_full_js(result.optimized_plan).position.cpu().numpy()[:, :, :10])
            cmd_idx = 0
            task_finish = False
            last_success_target = cube_position
        print(f"result: {result.success} status:{result.status} time:{result.total_time}")
        print(f"motion plan time: {time.time() - tt}")
        
        if not result.success[0]:
            return 0.05
    return 3.0  # 每秒调用一次


# region Main
bproc.init()

# load scene
# objs = bproc.loader.load_blend(
#     path="/ssd/yangyuqiang/infinigen/outputs/multi_dataset_no_plantandshlefobj/14ec7b18/fine/scene.blend",
#     obj_types=['mesh', 'curve', 'hair', 'armature','empty', 'light', 'camera'],
#     data_blocks=['armatures', 'cameras', 'collections', 'curves', 'images', 'lights', 'materials', 'meshes', 'objects', 'textures'])
objs = bproc.loader.load_blend(
    path="/ssd/yangyuqiang/infinigen/outputs/multi_dataset_no_plantandshlefobj/14ec7b18/fine/scene.blend",
    obj_types=['mesh', 'armature', 'camera'],
    data_blocks=['armatures', 'cameras', 'collections', 'meshes', 'objects'])
hide_collection("unique_assets:room_exterior")
hide_collection("unique_assets:room_ceiling")
bpy.context.window.workspace = bpy.data.workspaces["yuqiang"]
bpy.context.view_layer.update()

# load robot

robots = []
tensor_args = TensorDeviceType()

urdf_file="/ssd/yangyuqiang/curobo/src/curobo/content/assets/robot/ridgeback_franka/RidgebackFranka.urdf"
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(get_world_config, objs)]
    
    for i in range(NUM_ROBOTS):
        robot = bproc.loader.load_urdf(urdf_file="/ssd/yangyuqiang/curobo/src/curobo/content/assets/robot/ridgeback_franka/RidgebackFranka.urdf" + str(i))
        print("load success")
        robots.append(robot)

    for future in concurrent.futures.as_completed(futures):
        curobo_world_config = future.result()
    

# get start and desired pose
start_poses, end_poses = get_start_and_goal(objs)

init_arm_pose = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0 , 0.0])
blender_state = np.concatenate([start_poses[0] - 0.5, [np.pi/2], init_arm_pose])
set_robots_state(robots, blender_state)

# region Curobo
setup_curobo_logger("warn")
n_obstacle_mesh = 400
n_obstacle_cuboids = 50
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
max_attempts = 4
interpolation_dt = 0.05
enable_finetune_trajopt = False             

motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_cfg,
    world_cfg,
    tensor_args,
    collision_checker_type=CollisionCheckerType.MESH,
    num_trajopt_seeds=12,
    num_graph_seeds=1,
    interpolation_dt=interpolation_dt,
    collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
    optimize_dt=optimize_dt,
    trajopt_dt=trajopt_dt,
    trajopt_tsteps=trajopt_tsteps,
    trim_steps=trim_steps,
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, batch=NUM_ROBOTS) 
print(f"curobot is ready")


plan_config = MotionGenPlanConfig(
    enable_graph=False,
    enable_graph_attempt=2,
    max_attempts=max_attempts,
    enable_finetune_trajopt=enable_finetune_trajopt,
)

# add the target cube
curobo_joint_state = JointState.from_position(tensor_args.to_device(curobo_state),joint_names=motion_gen.rollout_fn.joint_names)
goal_state = motion_gen.rollout_fn.compute_kinematics(curobo_joint_state)
ee_pose = Pose(goal_state.ee_pos_seq, quaternion=goal_state.ee_quat_seq)
cube = bproc.object.create_primitive("CUBE", scale=[0.05, 0.05, 0.05], location=ee_pose.position[0].cpu().numpy())

collision_supported_world = WorldConfig.create_collision_support_world(curobo_world_config)
collision_supported_world.save_world_as_mesh("debug_collision_mesh.obj")

# region Timerg
print("start update world")
motion_gen.update_world(curobo_world_config.clone())
print("finish update world")
timer2 = bpy.app.timers.register(motion_plan_callback)

# while True:
#     bpy.context.view_layer.update()


