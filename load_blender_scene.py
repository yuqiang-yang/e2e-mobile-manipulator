import blenderproc as bproc
import numpy as np
import bpy

from blenderproc.python.types.MeshObjectUtility import MeshObject, create_primitive
from blenderproc.python.types.URDFUtility import URDFObject

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
    
def set_robot_pos(robot : URDFObject, joints_angle : list):
    for i, link in enumerate(robot.get_links_with_revolute_joints()):
        robot.set_rotation_euler_fk(link, rotation_euler=joints_angle[i], mode='absolute')
        
def update_view():
    bpy.context.view_layer.update()
    return 0.1 

def update_callback():
    print("update is running...")
    bpy.context.view_layer.update()
    return 0.2  # 每秒调用一次

cnt = 0
def motion_plan_callback():
    global cnt
    print("Motion plan is running...")
    set_robot_pos(robot, init_pose + cnt * 0.1)
    cnt += 1
    return 0.2  # 每秒调用一次
bproc.init()

# load scene
objs = bproc.loader.load_blend(
    path="/ssd/yangyuqiang/infinigen/outputs/multi_dataset_no_plantandshlefobj/14ec7b18/fine/scene.blend",
    obj_types=['mesh', 'curve', 'hair', 'armature','empty', 'light', 'camera'],
    data_blocks=['armatures', 'cameras', 'collections', 'curves', 'images', 'lights', 'materials', 'meshes', 'objects', 'textures'])

hide_collection("unique_assets:room_exterior")
hide_collection("unique_assets:room_ceiling")

# load robot
robot = bproc.loader.load_urdf(urdf_file="/ssd/yangyuqiang/curobo/src/curobo/content/assets/robot/ridgeback_franka/RidgebackFranka.urdf")

# get start and desired pose
start_poses, end_poses = get_start_and_goal(objs)
robot.set_location(start_poses[0])
init_pose = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0 , 0.0])
set_robot_pos(robot, init_pose)

timer1 = bpy.app.timers.register(update_callback)
timer2 = bpy.app.timers.register(motion_plan_callback)

# while True:
#     bpy.context.view_layer.update()
# import ipdb; ipdb.set_trace()


