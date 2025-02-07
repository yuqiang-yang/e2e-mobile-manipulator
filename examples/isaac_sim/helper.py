#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Standard Library
from typing import Dict, List

# Third Party
import numpy as np
from matplotlib import cm
from omni.isaac.core import World
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.objects import cuboid
from omni.isaac.core.robots import Robot
from pxr import UsdPhysics, UsdGeom

# CuRobo
from curobo.util.logger import log_warn
from curobo.util.usd_helper import set_prim_transform

ISAAC_SIM_23 = False
try:
    # Third Party
    from omni.isaac.urdf import _urdf  # isaacsim 2022.2
except ImportError:
    # Third Party
    from omni.importer.urdf import _urdf  # isaac sim 2023.1 or above

    ISAAC_SIM_23 = True
# Standard Library
from typing import Optional

# Third Party
from omni.isaac.core.utils.extensions import enable_extension

# CuRobo
from curobo.util_file import get_assets_path, get_filename, get_path_of_dir, join_path


def add_extensions(simulation_app, headless_mode: Optional[str] = None):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
    ]
    if headless_mode is not None:
        log_warn("Running in headless mode: " + headless_mode)
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    simulation_app.update()

    return True


############################################################
def add_robot_to_scene(
    robot_config: Dict,
    my_world: World,
    load_from_usd: bool = False,
    subroot: str = "",
    robot_name: str = "robot",
    position: np.array = np.array([0, 0, 0]),
):

    urdf_interface = _urdf.acquire_urdf_interface()
    # Set the settings in the import config
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751
    import_config.default_position_drive_damping = 52.35988
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    asset_path = get_assets_path()
    if (
        "external_asset_path" in robot_config["kinematics"]
        and robot_config["kinematics"]["external_asset_path"] is not None
    ):
        asset_path = robot_config["kinematics"]["external_asset_path"]
    full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
    dest_path = subroot
    robot_path = urdf_interface.import_robot(
        robot_path,
        filename,
        imported_robot,
        import_config,
        dest_path,
    )

    base_link_name = robot_config["kinematics"]["base_link"]

    robot_p = Robot(
        prim_path=robot_path + "/" + base_link_name,
        name=robot_name,
    )

    robot_prim = robot_p.prim
    stage = robot_prim.GetStage()
    linkp = stage.GetPrimAtPath(robot_path)
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    return robot, robot_path


class VoxelManager:
    def __init__(
        self,
        num_voxels: int = 5000,
        size: float = 0.02,
        color: List[float] = [1, 1, 1],
        prefix_path: str = "/World/curobo/voxel_",
        material_path: str = "/World/looks/v_",
    ) -> None:
        self.cuboid_list = []
        self.cuboid_material_list = []
        self.disable_idx = num_voxels
        for i in range(num_voxels):
            target_material = OmniPBR("/World/looks/v_" + str(i), color=np.ravel(color))

            cube = cuboid.VisualCuboid(
                prefix_path + str(i),
                position=np.array([0, 0, -10]),
                orientation=np.array([1, 0, 0, 0]),
                size=size,
                visual_material=target_material,
            )
            self.cuboid_list.append(cube)
            self.cuboid_material_list.append(target_material)
            cube.set_visibility(True)

    def update_voxels(self, voxel_position: np.ndarray, color_axis: int = 0):
        max_index = min(voxel_position.shape[0], len(self.cuboid_list))

        jet = cm.get_cmap("hot")  # .reversed()
        z_val = voxel_position[:, 0]

        jet_colors = jet(z_val)

        for i in range(max_index):
            self.cuboid_list[i].set_visibility(True)

            self.cuboid_list[i].set_local_pose(translation=voxel_position[i])
            self.cuboid_material_list[i].set_color(jet_colors[i][:3])

        for i in range(max_index, len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))

            # self.cuboid_list[i].set_visibility(False)

    def clear(self):
        for i in range(len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))
            
            
import random
import math
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf

def _set_random_color(prim):
    r = random.uniform(0, 1)
    g = random.uniform(0, 1)
    b = random.uniform(0, 1)
    color = (r, g, b)

    metallic = random.uniform(0, 1)
    roughness = random.uniform(0, 1)
    material_binding_api = UsdShade.MaterialBindingAPI(prim)
    material_path = prim.GetPath().AppendChild("Material")
    material = UsdShade.Material.Define(prim.GetStage(), material_path)
    shader = UsdShade.Shader.Define(prim.GetStage(), material_path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    material_binding_api.Bind(material)

def _generate_trajectory():
    amplitude = random.uniform(0.5, 2.0)
    angle = random.uniform(0, 2 * math.pi)
    period = random.randint(200, 500)
    return amplitude, angle, period

def calculate_position_offset(step, amplitude, angle, period):
    phase = 2 * math.pi * (step % period) / period
    dx = amplitude * math.sin(phase) * math.cos(angle)
    dy = amplitude * math.sin(phase) * math.sin(angle)
    dz = 0  
    return dx, dy, dz

def add_random_objects(my_world, num_cylinders=5, num_spheres=5, obs_range = 5,dynamic=True):
    
    random.seed(2)
    stage = my_world.stage

    # Function to create a cylinder
    def create_cylinder(stage, path, radius=0.5, height=2.0):
        cylinder_prim = stage.DefinePrim(path, "Cylinder")
        cylinder_geom = UsdGeom.Cylinder(cylinder_prim)
        cylinder_geom.GetRadiusAttr().Set(radius)
        cylinder_geom.GetHeightAttr().Set(height)
        return cylinder_prim

    # Function to create a sphere
    def create_sphere(stage, path, radius=0.5):
        sphere_prim = stage.DefinePrim(path, "Sphere")
        sphere_geom = UsdGeom.Sphere(sphere_prim)
        sphere_geom.GetRadiusAttr().Set(radius)
        return sphere_prim

    # Function to add collision properties
    def add_collision(prim):
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        return collision_api

    prims_with_trajectories = []

    # Add random cylinders
    for i in range(num_cylinders):
        x = random.uniform(-obs_range, obs_range)
        y = random.uniform(-obs_range, obs_range)
        z = random.uniform(0, 0.5)
        if np.linalg.norm([x, y]) < 1.0:
            continue
        radius = random.uniform(0.2, 0.6)
        path = f"/World/Cylinder_{i}"
        cylinder_prim = create_cylinder(stage, path, radius)
        UsdGeom.XformCommonAPI(cylinder_prim).SetTranslate((x, y, z))
        # add_collision(cylinder_prim)
        _set_random_color(cylinder_prim)
        amplitude, angle, period = _generate_trajectory()
        amplitude = amplitude if dynamic else 0.0
        prims_with_trajectories.append((cylinder_prim, (x, y, z), amplitude, angle, period))

    # Add random spheres
    for i in range(num_spheres):
        x = random.uniform(-obs_range, obs_range)
        y = random.uniform(-obs_range, obs_range)
        z = random.uniform(1, 3)  # Suspended in the air
        if np.linalg.norm([x, y]) < 1.0:
            continue
        radius = random.uniform(0.2, 0.6)
        path = f"/World/Sphere_{i}"
        sphere_prim = create_sphere(stage, path, radius)
        UsdGeom.XformCommonAPI(sphere_prim).SetTranslate((x, y, z))
        # add_collision(sphere_prim)
        _set_random_color(sphere_prim)
        amplitude, angle, period = _generate_trajectory()
        amplitude = amplitude if dynamic else 0.0
        prims_with_trajectories.append((sphere_prim, (x, y, z), amplitude, angle, period))

    print(f"Added {num_cylinders} cylinders and {num_spheres} spheres to the world.")
    return prims_with_trajectories
