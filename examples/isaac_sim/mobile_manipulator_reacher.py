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
NUM_ROBOTS = 4
ready_to_plan = False
cmd_plan = None
cube_position = np.zeros(3)
cube_orientation = None
tensor_args = None
motion_gen = None
cu_js = None
plan_config = None
sim_js_names = None
robot = None
target_pos = np.zeros((3, 3))
last_success_target = np.zeros(3)
success_result = None
cmd_idx = 0
############################################################

def replan_thread():
    global cmd_plan, cmd_idx, target_pos, success_result, last_success_target
    while True:
        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()
        target_pos[-1] = cube_position.copy() 
        if np.linalg.norm(last_success_target - cube_position) > 0.1 \
            and np.linalg.norm(target_pos[2] - target_pos[0]) == 0.0 \
            and ready_to_plan:
            target_changed = np.linalg.norm(last_success_target - cube_position) > 0.1
            cube_position, cube_orientation = target.get_world_pose()
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation
            # compute curobo solution:
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            print(f"cu_js.shape {cu_js.shape} ik_goal:{ik_goal.position}")
            if obstacles_config is not None:
                motion_gen.update_world(obstacles_config)
                
            result = motion_gen.plan_batch(
                cu_js.clone(),
                ik_goal.clone().repeat_seeds(NUM_ROBOTS),
                plan_config.clone(),
                ik_seeds=success_result.optimized_plan.position[:, -1].unsqueeze(1) if target_changed and success_result is not None else None,
                trajopt_seeds=success_result.optimized_seeds if target_changed and success_result is not None else None 
            )
            # import ipdb; ipdb.set_trace()
            succ = result.success[0].item()  # ik_result.success.item()
            print(f"result ik time:{result.ik_time:.3f} graph time:{result.graph_time:.3f} " \
                    f"opt_time:{result.trajopt_time:.3f} finetune:{result.finetune_time:.3f} total:{result.total_time:.3f} " \
                    f"attemps:{result.attempts} opt_attemps:{result.trajopt_attempts}")
            if succ:
                # cmd_plan = result.get_interpolated_plan()
                cmd_plan = result.optimized_plan
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                # get only joint names that are in both:
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                cmd_idx = 0
                last_success_target = cube_position
                success_result = result.clone()
                time.sleep(0.5)
            else:
                carb.log_warn("Plan did not converge to a solution: " + str(result.status))      
        else:
            time.sleep(0.1)
        target_pos[0] = target_pos[1]
        target_pos[1] = target_pos[2]
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
    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt
        enable_finetune_trajopt = False

    # motion_gen_config = MotionGenConfig.load_from_robot_config(robot_cfg, world_cfg, tensor_args, collision_checker_type=CollisionCheckerType.MESH, num_trajopt_seeds=12, num_graph_seeds=12, interpolation_dt=interpolation_dt, collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh}, optimize_dt=optimize_dt, trajopt_dt=trajopt_dt, trajopt_tsteps=trajopt_tsteps, trim_steps=trim_steps)
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
    if not args.reactive:
        print("warming up...")
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, batch=NUM_ROBOTS)

    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
    )

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
    prims_with_trajectories = add_random_objects(my_world, 50, 50, obs_range=10, dynamic=False)
    ##### Isaac custom config #######
    thread = threading.Thread(target=replan_thread)
    thread.daemon = True
    thread.start()
    obstacles_config = None
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index

        if obstacles_config is not None:
            name2idx = {}
            for i, mesh in enumerate(obstacles_config.mesh):
                name2idx[mesh.name] = i
        # update dynamic obstacle pos
        for prim, initial_position, amplitude, angle, period in prims_with_trajectories:
            dx, dy, dz = calculate_position_offset(step_index, amplitude, angle, period)
            new_x = initial_position[0] + dx
            new_y = initial_position[1] + dy
            new_z = initial_position[2] + dz
            UsdGeom.XformCommonAPI(prim).SetTranslate((new_x, new_y, new_z))
            if obstacles_config is not None:
                obstacles_config.mesh[name2idx["/World/" + prim.GetName()]].pose = [new_x, new_y, new_z, 1, 0, 0, 0]
                
        if step_index < 2:
            my_world.reset()
            for robot in robots:
                robot._articulation_view.initialize()
                idx_list = [robot.get_dof_index(x) for x in j_names]
                robot.set_joint_positions(default_config, idx_list)
                robot._articulation_view.set_max_efforts(
                    values=np.array([50 for i in range(len(idx_list))]), joint_indices=idx_list
                )
        if step_index < 20:
            continue

        if True:
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
                
            # pt1 = time.time()
            
            # print(f"Updating world, obj num:{len(obstacles_config.objects)} load time:{pt1 - start} total_time {time.time() - start}")

        
        sim_js_pos =  []
        sim_js_vel = []
        for robot in robots:
            sim_js = robot.get_joints_state()
            sim_js_pos.append(sim_js.positions)
            sim_js_vel.append(sim_js.velocities)
            sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js_local = JointState(
            position=tensor_args.to_device(torch.Tensor(sim_js_pos)),
            velocity=tensor_args.to_device(torch.Tensor(sim_js_vel)),  # * 0.0,
            acceleration=tensor_args.to_device(torch.Tensor(sim_js_vel)) * 0.0,
            jerk=tensor_args.to_device(torch.Tensor(sim_js_vel)) * 0.0,
            joint_names=sim_js_names,
        )

        if not args.reactive:
            cu_js_local.velocity *= 0.0
            cu_js_local.acceleration *= 0.0

        # if args.reactive and past_cmd is not None:
        #     cu_js.position[:] = past_cmd.position
        #     cu_js.velocity[:] = past_cmd.velocity
        #     cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js_local.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:

            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position[0])
            if spheres is None:
                spheres = []

                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
        ready_to_plan = True
        if cmd_plan is not None:
            DOFS = len(robots[0].dof_names)
            cmd_state_tensor = cmd_plan.get_state_tensor()[:,cmd_idx]
            past_cmd = cmd_state_tensor.clone()
            for i, robot in enumerate(robots):
                robot.set_joint_positions(cmd_state_tensor[i, :DOFS].cpu().numpy())
            
            if step_index % 2 == 0:
                cmd_idx += 1

            # print(f"cmd idx:{cmd_idx}/{len(cmd_plan.position[0])} vel:{cmd_state_tensor[0, DOFS:DOFS+2]}")
            if cmd_idx >= len(cmd_plan.position[0]):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
    simulation_app.close()


# if __name__ == "__main__":
#     main()
