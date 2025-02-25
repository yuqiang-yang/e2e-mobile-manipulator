import blenderproc as bproc
import numpy as np
import bpy
import torch
import argparse

from typing import List 

from blenderproc.python.types.MeshObjectUtility import MeshObject, create_primitive
from blenderproc.python.types.URDFUtility import URDFObject

#curobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult
)


#####################Global Variables#######################
NUM_ROBOTS = 2
cube_position = np.zeros(3)
cube_orientation = None
target_pos = np.zeros((3, 3))
last_success_target = np.zeros(3)
success_result = None
############################################################

def hide_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection:
        collection.hide_viewport = True
        collection.hide_render = True
        collection.hide_select = True

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
    
def set_robots_state(robots :  List[URDFObject], state : np.ndarray):
    BASE_SHIFT = 4
    if len(state.shape) == 1:
        state = np.tile(state, (NUM_ROBOTS, 1))

    for i, robot in enumerate(robots):
        for j, link in enumerate(robot.get_links_with_revolute_joints()):
            robot.set_location(state[i][:3]) #x y z
            robot.set_rotation_euler([0, 0, state[i][3]]) #yaw
            robot.set_rotation_euler_fk(link, rotation_euler=state[i][j+BASE_SHIFT], mode='absolute')
        
def update_callback():
    print("update is running...")
    bpy.context.view_layer.update()
    return 0.2  # 每秒调用一次

cnt = 0
def motion_plan_callback():
    global cnt
    print("Motion plan is running...")
    set_robots_state(robots, init_pose + cnt * 0.1)
    cnt += 1
    return 3.0  # 每秒调用一次
bproc.init()

# load scene
objs = bproc.loader.load_blend(
    path="/ssd/yangyuqiang/infinigen/outputs/multi_dataset_no_plantandshlefobj/14ec7b18/fine/scene.blend",
    obj_types=['mesh', 'curve', 'hair', 'armature','empty', 'light', 'camera'],
    data_blocks=['armatures', 'cameras', 'collections', 'curves', 'images', 'lights', 'materials', 'meshes', 'objects', 'textures'])

hide_collection("unique_assets:room_exterior")
hide_collection("unique_assets:room_ceiling")

# load robot
robots = []
for i in range(NUM_ROBOTS):
    robot = bproc.loader.load_urdf(urdf_file="/ssd/yangyuqiang/curobo/src/curobo/content/assets/robot/ridgeback_franka/RidgebackFranka.urdf" + str(i))
    robots.append(robot)

# get start and desired pose
start_poses, end_poses = get_start_and_goal(objs)

init_arm_pose = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0 , 0.0])
set_robots_state(robots, np.concatenate([start_poses[0], [np.pi/2], init_arm_pose]))

# init curobo
setup_curobo_logger("warn")
n_obstacle_mesh = 100
n_obstacle_cuboids = 50
tensor_args = TensorDeviceType()
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


# timer1 = bpy.app.timers.register(update_callback)
timer2 = bpy.app.timers.register(motion_plan_callback)

# while True:
#     bpy.context.view_layer.update()


