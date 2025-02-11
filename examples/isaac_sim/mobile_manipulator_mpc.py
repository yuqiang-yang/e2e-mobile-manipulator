import isaacsim
import torch
import argparse
import time
import threading

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument("--robot", type=str, default="ridgeback_franka.yml", help="robot configuration to load")
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

args = parser.parse_args()

############################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
# Standard Library
from typing import Dict

# Third Party
import carb
import numpy as np
from helper import *
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction
import omni.kit.actions.core

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
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
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

from traj_server import *



def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [
            (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
        ]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)



#####################Global Variables#######################
NUM_ROBOTS = 1
PLAN_AHEAD_TIME = 0.15
ready_to_plan = False
cube_position = np.zeros(3)
cube_orientation = None
tensor_args = None
mpc = None
traj_server = TrajServer()
sim_js_names = None
robot = None
target_pos = np.zeros((3, 3))
last_success_target = np.zeros(3)
success_result = None
replan = True
cmd_action = None
############################################################

def replan_thread():
    global target_pos, success_result, last_success_target, replan, cmd_action

if __name__ == "__main__":
    # create a curobo motion gen instance:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.75]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 200
    n_obstacle_mesh = 100

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robots, robot_prim_paths = add_multiple_robots(NUM_ROBOTS, robot_cfg, my_world, subroot="/Robot",\
        positions=np.array([(0.0 * i, 0, 0) for i in range(NUM_ROBOTS)]))
    robot, robot_prim_path = robots[0], robot_prim_paths[0]

    world_cfg = WorldConfig()

    trajopt_dt = None
    optimize_dt = False
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4
    interpolation_dt = 0.05
    enable_finetune_trajopt = False

    # motion_gen_config = MotionGenConfig.load_from_robot_config(robot_cfg, world_cfg, tensor_args, collision_checker_type=CollisionCheckerType.MESH, num_trajopt_seeds=12, num_graph_seeds=12, interpolation_dt=interpolation_dt, collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh}, optimize_dt=optimize_dt, trajopt_dt=trajopt_dt, trajopt_tsteps=trajopt_tsteps, trim_steps=trim_steps)
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=True,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.02,
    )
    
    mpc = MpcSolver(mpc_config)

    # retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0).repeat(NUM_ROBOTS, 1)
    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone()
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
    )
    # goal_buffer = mpc.setup_solve_batch(goal, 1)
    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=2)
    
    print("Curobo is Ready")
    add_extensions(simulation_app, args.headless_mode)

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None
    step_index = 0

    ##### Isaac custom config #######
    action_registry = omni.kit.actions.core.get_action_registry()

    # switches to camera lighting
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
    action.execute()
    # prims_with_trajectories = add_random_objects(my_world, 50, 50, obs_range=10, dynamic=False)
    ##### Isaac custom config #######
    thread = threading.Thread(target=replan_thread)
    thread.daemon = True
    thread.start()
    obstacles_config = None
    init_world = False
    cmd_state_full = None

    while simulation_app.is_running():
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
        draw_points(mpc.get_visual_rollouts())
        
        my_world.step(render=True)
        
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index

        # if obstacles_config is not None:
        #     name2idx = {}
        #     for i, mesh in enumerate(obstacles_config.mesh):
        #         name2idx[mesh.name] = i
        # update dynamic obstacle pos
        # for prim, initial_position, amplitude, angle, period in prims_with_trajectories:
        #     dx, dy, dz = calculate_position_offset(step_index, amplitude, angle, period)
        #     new_x = initial_position[0] + dx
        #     new_y = initial_position[1] + dy
        #     new_z = initial_position[2] + dz
        #     UsdGeom.XformCommonAPI(prim).SetTranslate((new_x, new_y, new_z))
        #     if obstacles_config is not None:
        #         obstacles_config.mesh[name2idx["/World/" + prim.GetName()]].pose = [new_x, new_y, new_z, 1, 0, 0, 0]

        if step_index <= 2:
            my_world.reset()
            for robot in robots:
                robot._articulation_view.initialize()
                idx_list = [robot.get_dof_index(x) for x in j_names]
                robot.set_joint_positions(default_config, idx_list)
                robot._articulation_view.set_max_efforts(
                    values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
                )
        if step_index < 20:
            continue

        if step_index % 1000 == 0:
            start = time.time()
            if obstacles_config is None:
                obstacles_config = usd_help.get_obstacles_from_stage(
                    # only_paths=[obstacles_path],
                    reference_prim_path=robot_prim_path,
                    ignore_substring=[
                        robot_prim_path,
                        "/World/target",
                        "/World/defaultGroundPlane",
                        "/curobo",
                        "/Ridge"
                    ],
                ).get_collision_check_world()
                mpc.update_world(obstacles_config.clone())

        sim_js_pos =  []
        sim_js_vel = []
        for robot in robots:
            sim_js = robot.get_joints_state()
            sim_js_pos.append(sim_js.positions)
            sim_js_vel.append(sim_js.velocities)
            sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        ready_to_plan = True

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()
        if ready_to_plan:
            target_unchanged = np.linalg.norm(last_success_target - cube_position) < 0.1
            cube_position, cube_orientation = target.get_world_pose()
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation
            # compute curobo solution:
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            # print(f"ik pose {ik_goal.position}")
            # Get plan state
            plan_time = time.time()
            cu_js = JointState(
                position=tensor_args.to_device(torch.Tensor(sim_js_pos[0])),
                velocity=tensor_args.to_device(torch.Tensor(sim_js_vel[0]))* 0.0,
                acceleration=tensor_args.to_device(torch.Tensor(sim_js_vel[0])) * 0.0,
                jerk=tensor_args.to_device(torch.Tensor(sim_js_vel[0])) * 0.0,
                joint_names=sim_js_names,
            )

            cu_js = cu_js.get_ordered_joint_state(mpc.kinematics.joint_names)
            
            if cmd_state_full is None:
                current_state.copy_(cu_js)
            else:
                current_state_partial = cmd_state_full.get_ordered_joint_state(
                    mpc.rollout_fn.joint_names
                )
                current_state.copy_(current_state_partial)
                current_state.joint_names = current_state_partial.joint_names
                # current_state = current_state.get_ordered_joint_state(mpc.rollout_fn.joint_names)
            current_state.copy_(cu_js)
            goal_buffer.goal_pose.copy_(ik_goal)
            mpc.update_goal(goal_buffer)
            current_state.copy_(cu_js)
            mpc_result = mpc.step(current_state, max_attempts=2)
            print("solver time",mpc_result.solve_time)
            # import ipdb; ipdb.set_trace()
            succ = mpc_result.metrics.feasible[0].item()  # ik_result.success.item()
            print(f"mpc success:{succ}. sover time {mpc_result.solve_time}")
            if succ:
                cmd_action = mpc_result.js_action                
                last_success_target = cube_position
                success_result = mpc_result.clone()
                print(f"res {cmd_action.position.cpu().numpy()}")

                time.sleep(0.05)
            else:
                pass
                # carb.log_warn("Plan did not converge to a solution: " + str(result.status))
        else:
            time.sleep(0.1)
        
        if traj_server.ready():
            articulation_controller = robot.get_articulation_controller()
            cmd_state_full = mpc_result.js_action
            common_js_names = []
            idx_list = []
            for x in sim_js_names:
                if x in cmd_state_full.joint_names:
                    idx_list.append(robot.get_dof_index(x))
                    common_js_names.append(x)

            cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
            cmd_state_full = cmd_state
            art_action = ArticulationAction(
            cmd_state.position[0].cpu().numpy(),
            # cmd_state.velocity.cpu().numpy(),
            joint_indices=idx_list,
        )
            DOFS = len(robots[0].dof_names)
            # for i, robot in enumerate(robots):
            #     robot.set_joint_positions(cmd_action.position[i].cpu().numpy())
            for _ in range(3):
                articulation_controller.apply_action(art_action)
    simulation_app.close()


# if __name__ == "__main__":
#     main()
