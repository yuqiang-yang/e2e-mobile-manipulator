import blenderproc as bproc
import sys
import argparse
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import cv2
import numpy as np
import random
import trimesh
import open3d as o3d
from tqdm import tqdm
import json
import bpy
import matplotlib.pyplot as plt
import imageio
import shutil
from path_utils.path_planner import PathPlanner
from path_utils.geometry_tools import *

def replace_all_lights():
    # Iterate through all objects in the scene
    for obj in bpy.context.scene.objects:
        # Check if the object is a light
        if obj.type == 'LIGHT':
            obj.data.color = np.random.uniform([0.1, 0.1, 0.1], [1, 1, 1])
            obj.data.energy = np.random.uniform(10,100)
    bpy.context.view_layer.update()

def randomize_lights(path_points,ceiling_height=1.8):
    num_lights = path_points.shape[0]
    light_indexes = np.linspace(0,path_points.shape[0]-1,num_lights).astype(np.int32)
    for _,index in enumerate(light_indexes):
        light_pos = path_points[index]
        light_pos[2] = light_pos[2] + ceiling_height
        light_color = np.random.uniform([0.9, 0.9, 0.9], [1, 1, 1])
        light_energy = np.random.uniform(20, 100)
        # light_energy = np.random.uniform(500, 1000)
        # Create a new light object
        # light_data = bpy.data.lights.new(name="RandomLight_%d"%index, type='SPOT')
        light_data = bpy.data.lights.new(name="RandomLight_%d"%index, type='POINT')
        light_object = bpy.data.objects.new(name="RandomLight_%d"%index, object_data=light_data)
        # Set light properties
        light_object.location = light_pos
        light_data.color = light_color
        light_data.energy = light_energy
        # light_data.spot_size = np.radians(360) # Randomize cone angle
        # light_data.spot_size = 100 # Randomize cone angle
        # light_data.spot_blend = np.random.uniform(0, 1)
        # Link the light to the scene
        bpy.context.collection.objects.link(light_object)

def set_global_illumination():
    # 创建一个新的世界设置
    world = bpy.data.worlds.new("GlobalIllumination")
    # 设置世界为当前场景的活跃世界
    bpy.context.scene.world = world
    # 确保世界节点树不为空
    if world.node_tree is None:
        world.use_nodes = True
    # 添加背景节点
    if "Background" not in world.node_tree.nodes:
        bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
    else:
        bg_node = world.node_tree.nodes["Background"]
    # 设置环境光照的颜色
    bg_node.inputs[0].default_value = (1, 1, 1, 1)
    # 设置环境光照的强度
    bg_node.inputs[1].default_value = 1.0
    # bproc.object.set_category_id("GlobalIllumination", None)
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"], default_values={'category_id': None})



parser = argparse.ArgumentParser()
parser.add_argument("--scene_index",type=int,default=1)
parser.add_argument("--gpu_id",type=int,default=0)
# parser.add_argument("--blender", help="Path to the blender file", default='/home/pengjiaqi/infinigen/outputs/indoors/diningroom/')
parser.add_argument("--blender", help="Path to the blender file", default='/ssd/yangyuqiang/infinigen/outputs/multi_dataset_clutter/')
# parser.add_argument("--blender", help="Path to the blender file", default='/home/pengjiaqi/infinigen/outputs/multi_dataset_clutter/56ba3798/coarse/')
parser.add_argument("--output_dir", help="Path to where the data should be saved",default="/home/yangyuqiang/infinigen/navdata_engine/nav_datasets/multi_dataset_clutter/")
parser.add_argument("--image_height",type=int,default=180)
parser.add_argument("--image_width",type=int,default=320)
parser.add_argument("--camera_hfov",type=float,default=68)
parser.add_argument("--camera_vfov",type=float,default=42)
parser.add_argument("--ceiling_height",type=float,default=1.8)
parser.add_argument("--safe_distance",type=float,default=0.15)
args = parser.parse_known_args()[0]
if not os.path.exists(args.blender):
    raise Exception("One of the two folders does not exist!")
# blenderproc initialization
bproc.init()
bproc.renderer.set_render_devices(False,"CUDA",[args.gpu_id])
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

for root, dirs, files in os.walk(args.blender):
    for file in files:
        if not file.endswith('.blend'):
            continue

        blender_path = os.path.join(root, file)
        relative_path = os.path.relpath(blender_path, args.blender)
        house_id = '_'.join(relative_path.split('/')[:-1])
        print("Processing %s"%house_id)
        # import pdb; pdb.set_trace()
        bproc.renderer.set_max_amount_of_samples(128)

        objs = bproc.loader.load_blend(
            path=blender_path,
            obj_types=['mesh', 'curve', 'hair', 'armature','empty', 'light', 'camera'],
            data_blocks=['armatures', 'cameras', 'collections', 'curves', 'images', 'lights', 'materials', 'meshes', 'objects', 'textures'])

        loaded_objects = bproc.object.get_all_mesh_objects()
        print("Loaded %d objects"%len(loaded_objects))

        scene_pcd = o3d.geometry.PointCloud()
        num_loaded_objects = 0
        for index,obj in enumerate(loaded_objects):
            if not isinstance(obj,bproc.types.MeshObject):
                continue
            bpy.context.view_layer.objects.active = obj.blender_obj
            obj_mesh = obj.mesh_as_trimesh()

            # 检查 obj_mesh.faces 是否为空
            if obj_mesh.faces.shape[0] == 0:
                bpy.context.view_layer.objects.active = None  # 关闭活动对象
                continue

            obj_mesh = obj_mesh.apply_transform(obj.get_local2world_mat())
            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices = o3d.utility.Vector3dVector(obj_mesh.vertices)
            new_mesh.triangles = o3d.utility.Vector3iVector(obj_mesh.faces)
            # new_mesh.vertex_normals = o3d.utility.Vector3dVector(obj_mesh.vertex_normals)
            new_mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(obj_mesh.vertex_normals, copy=True))
            new_pcd = new_mesh.sample_points_uniformly(200000)
            new_pcd = new_pcd.voxel_down_sample(0.05)
            new_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(np.array(new_pcd.points)) * 0.4)
            scene_pcd = scene_pcd + new_pcd
            scene_pcd = scene_pcd.voxel_down_sample(0.05)
            num_loaded_objects += 1
        scene_points = np.array(scene_pcd.points)
        scene_pcd.points = o3d.utility.Vector3dVector(scene_points)

        points = np.array(scene_pcd.points)
        floor_heights = extract_floor_heights(points)
        path_planner = PathPlanner(ceiling_offset=args.ceiling_height,safe_distance=args.safe_distance)

        light_points = np.array(scene_pcd.voxel_down_sample(2.0).points)
        randomize_lights(light_points,args.ceiling_height)
        set_global_illumination()

        print("Extracted %d loaded_objects"%num_loaded_objects)
        print("Extracted %d floors"%len(floor_heights))
        for floor_index,current_floor in enumerate(floor_heights):
            current_floor = current_floor[0]
            try:
                path_planner.reset(current_floor,0.5,scene_pcd)
            except:
                continue
            for trajectory_num in tqdm(range(300)):
                trajectory_index = floor_index * 100 + trajectory_num
                # bproc.renderer.set_light_bounces(diffuse_bounces=np.random.randint(50,100), glossy_bounces=np.random.randint(50,600), max_bounces=np.random.randint(50,600),transmission_bounces=np.random.randint(50,600),transparent_max_bounces=np.random.randint(50,600))
                camera_height = np.random.uniform(0.25,1.25)
                camera_hfov = np.random.uniform(50, 100) # 68
                # camera_vfov = np.random.uniform(40, 90) # 42
                aspect_ratio = args.image_width / args.image_height
                camera_vfov = 2 * np.arctan(np.tan(np.radians(camera_hfov) / 2) / aspect_ratio) * (180 / np.pi)

                camera_intrinsic = generate_intrinsic(args.image_width,args.image_height,args.camera_hfov,args.camera_vfov)
                bpy.context.scene.frame_end = 0
                random_index = np.random.choice(np.nonzero(path_planner.safe_value > 0.1)[0])
                camera_translation = np.array(path_planner.navigable_pcd.points)[random_index]
                camera_translation[2] = camera_height + current_floor
                pitch_rad = np.deg2rad((210 - 180*camera_height))
                camera_rotation = [np.clip(pitch_rad,np.pi/3,np.pi/2),0,0]

                tobase_extrinsic = bproc.math.build_transformation_mat(camera_translation * np.array([0,0,1]),camera_rotation)
                distance = np.sum(np.abs(np.array(path_planner.navigable_pcd.points) - camera_translation),axis=-1)
                # condition = np.where((path_planner.safe_value>0.2) & (distance > 5.0))[0] # 大于一定的安全距离，且距离够远
                condition = np.where((path_planner.safe_value>0.1) & (distance > 7.0))[0]
                if condition.shape[0] == 0:
                    continue
                target_point = np.array(path_planner.navigable_pcd.points)[np.random.choice(condition)]
                status,waypoints,wayrotations = path_planner.generate_trajectory(camera_translation,target_point)
                if status == False:
                    continue
                waypoints = np.concatenate((waypoints,np.ones((waypoints.shape[0],1)) * camera_translation[2]),axis=-1)
                wayrotations = np.stack((np.array([camera_rotation[0]]*waypoints.shape[0]),np.zeros((waypoints.shape[0],)),wayrotations),axis=-1)
                path_pcd = cpu_pointcloud_from_array(waypoints,np.ones_like(waypoints) * np.array([1,0,0]))
                if waypoints.shape[0] > 500:
                    continue

                os.makedirs("%s/%s/trajectory_%d/"%(args.output_dir,house_id,trajectory_index),exist_ok=False)
                os.makedirs("%s/%s/trajectory_%d/rgb/"%(args.output_dir,house_id,trajectory_index),exist_ok=False)
                os.makedirs("%s/%s/trajectory_%d/depth/"%(args.output_dir,house_id,trajectory_index),exist_ok=False)
                o3d.io.write_point_cloud("%s/%s/trajectory_%d/path.ply"%(args.output_dir,house_id,trajectory_index),path_pcd+path_planner.navigable_pcd)
                o3d.io.write_point_cloud("%s/%s/trajectory_%d/esdf.ply"%(args.output_dir,house_id,trajectory_index),path_planner.esdf_pcd)
                cv2.imwrite("%s/%s/trajectory_%d/decision_map.jpg"%(args.output_dir,house_id,trajectory_index),path_planner.color_decision_map)

                fps_writer = cv2.VideoWriter("%s/%s/trajectory_%d/fps.mp4" % (args.output_dir, house_id, trajectory_index), 
                             cv2.VideoWriter_fourcc(*'mp4v'), 10, (args.image_width, args.image_height))
                depth_writer = cv2.VideoWriter("%s/%s/trajectory_%d/depth.mp4" % (args.output_dir, house_id, trajectory_index), 
                                            cv2.VideoWriter_fourcc(*'mp4v'), 10, (args.image_width, args.image_height))

                camera_trajectory = []
                for pt,rt in zip(waypoints,wayrotations):
                    camera_pose = bproc.math.build_transformation_mat(pt, rt)
                    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(pt, rt))
                    camera_trajectory.append(bproc.math.build_transformation_mat(pt, rt).tolist())

                bproc.camera.set_intrinsics_from_K_matrix(camera_intrinsic,args.image_width,args.image_height)
                data = bproc.renderer.render()

                save_flag = True
                for ci,color,depth in zip(np.arange(len(data['colors'])),data['colors'],data['depth']):
                    if np.where(color.sum(axis=-1)==0)[0].shape[0] > 5000 or np.where(depth < args.safe_distance)[0].shape[0] > 5000:
                        save_flag = False
                        break
                    # if np.where(depth < args.safe_distance)[0].shape[0] > 5000:
                    #     save_flag = False
                    #     break
                    fps_writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                    depth_normalized = (np.clip(depth / 5.0, 0, 1) * 255.0).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                    depth_writer.write(depth_colored)
                    cv2.imwrite("%s/%s/trajectory_%d/rgb/%d.jpg"%(args.output_dir,house_id,trajectory_index,ci),cv2.cvtColor(color,cv2.COLOR_BGR2RGB))
                    cv2.imwrite("%s/%s/trajectory_%d/depth/%d.png"%(args.output_dir,house_id,trajectory_index,ci),np.clip(depth * 10000.0,0,65535).astype(np.uint16))

                # fps_writer.close()
                # depth_writer.close()
                fps_writer.release()
                depth_writer.release()
                save_dict = {'camera_intrinsic':camera_intrinsic.tolist(),
                            'camera_extrinsic':tobase_extrinsic.tolist(),
                            'camera_trajectory':camera_trajectory}
                json_object = json.dumps(save_dict, indent=4)
                with open("%s/%s/trajectory_%d/data.json"%(args.output_dir,house_id,trajectory_index), "w") as outfile:
                    outfile.write(json_object)

                if not save_flag:
                    shutil.rmtree("%s/%s/trajectory_%d"%(args.output_dir,house_id,trajectory_index))