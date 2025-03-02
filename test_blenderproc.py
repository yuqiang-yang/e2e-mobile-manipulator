import blenderproc as bproc
import bpy
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--scene_path", type=str, default="/ssd/yangyuqiang/infinigen/outputs/multi_dataset_big_door/77f33467/fine/scene.blend")
#/ssd/yangyuqiang/infinigen/outputs/multi_dataset_big_door_less_obs/2a14347
parser.add_argument("--trajs", type=int, default=10)
parser.add_argument("--image_height",type=int,default=480)
parser.add_argument("--image_width",type=int,default=640)
parser.add_argument("--camera_hfov",type=float,default=87)
parser.add_argument("--camera_vfov",type=float,default=54)
parser.add_argument("--device",type=int,default=3)
args = parser.parse_args()

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

def generate_intrinsic(width,height,hfov,vfov):
    intrinsic = np.eye(3)
    intrinsic[0][0] = width / (2 * (np.tan(np.deg2rad(hfov)/2)))
    intrinsic[1][1] = height / (2 * (np.tan(np.deg2rad(vfov)/2)))
    intrinsic[0][2] = width / 2
    intrinsic[1][2] = height / 2
    return intrinsic

def setup_studio_light(studio_light_name, strength=1.0):
    """
    自动化设置 Studio Light 作为环境光，并应用到最终渲染。
    
    :param studio_light_name: Studio Light 的名称（不带扩展名）
    :param strength: 环境光的强度
    """
    # 获取 Blender 安装目录
    blender_data_path = bpy.utils.resource_path('LOCAL')
    studio_light_dir = os.path.join(blender_data_path, "datafiles", "studiolights", "world")
    
    # 构造 Studio Light 的完整路径
    studio_light_path = os.path.join(studio_light_dir, f"{studio_light_name}")
    
    if not os.path.exists(studio_light_path):
        raise FileNotFoundError(f"Studio Light 文件未找到: {studio_light_path}")
    
    # 加载 Studio Light 的 HDRI 文件
    world = bpy.context.scene.world
    world.use_nodes = True
    
    # 清空现有的 World 节点
    nodes = world.node_tree.nodes
    nodes.clear()
    
    # 添加 Environment Texture 节点
    env_texture = nodes.new(type="ShaderNodeTexEnvironment")
    env_texture.image = bpy.data.images.load(studio_light_path)
    
    # 添加 Background 节点
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Strength"].default_value = strength  # 设置环境光强度
    
    # 添加 Output 节点
    output = nodes.new(type="ShaderNodeOutputWorld")
    
    # 连接节点
    links = world.node_tree.links
    links.new(env_texture.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])

    print(f"已成功加载 Studio Light: {studio_light_name}")

    
# 初始化 BlenderProc
bproc.init()

# bproc.renderer.enable_depth_output(activate_antialiasing=False)
camera_intrinsic = generate_intrinsic(args.image_width,args.image_height,args.camera_hfov,args.camera_vfov)
bproc.camera.set_intrinsics_from_K_matrix(camera_intrinsic,args.image_width,args.image_height)

urdf_file = "/ssd/yangyuqiang/curobo/src/curobo/content/assets/robot/ridgeback_franka/RidgebackFranka.urdf"
bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
objs = bproc.loader.load_blend(
    "/ssd/yangyuqiang/infinigen/outputs/multi_dataset_big_door/77f33467/fine/scene.blend",
    obj_types=['mesh', 'curve', 'hair', 'armature', 'empty', 'light', 'camera'],
    data_blocks=['armatures', 'cameras', 'collections', 'curves', 'images', 'lights', 'materials', 'meshes', 'objects', 'textures'])
# if not bpy.context.scene.camera:
#     print("未找到相机，正在添加默认相机...")
#     cam_pose = bproc.math.build_transformation_mat(location=[0, -5, 0], rotation=[0, 0, 0])
#     bproc.camera.add_camera_pose(cam_pose)

robot = bproc.loader.load_urdf(urdf_file=urdf_file)

scene = bpy.context.scene

# 禁用场景中的灯光
# scene.eevee.use_gtao = False  # 禁用环境光遮蔽 (Ambient Occlusion)
# scene.eevee.use_bloom = False  # 禁用光晕效果
# scene.eevee.use_ssr = False  # 禁用屏幕空间反射 (Screen Space Reflections)
setup_studio_light("forest.exr", strength=1.0)
# set_global_illumination()
# for a in bpy.context.screen.areas:
#     if a.type == "VIEW_3D":
        
#         for s in a.spaces:
#             # s.shading.use_scene_lights = False
#             # s.shading.use_scene_world = False
#             # # s.shading.studio_light = "/ssd/yangyuqiang/miniforge3/envs/isaac-sim42/lib/python3.10/site-packages/bpy/4.0/datafiles/studiolights/world/forest.exr"
#             # import ipdb; ipdb.set_trace()
            
#             s.shading.type = 'RENDERED'
#             s.shading.use_scene_lights_render = False
#             s.shading.use_scene_world_render = False
#             s.shading.use_scene_world_render = False
#             s.shading.studio_light = 'forest.exr'
# 禁用世界环境光
# if scene.world:
#     scene.world.use_nodes = True  # 启用节点系统
#     nodes = scene.world.node_tree.nodes
#     for node in nodes:
#         if node.type == 'BACKGROUND':
#             node.inputs['Strength'].default_value = 0.0  # 将背景光强度设置为 0
# bproc.world.set_world_as_environment_map("/ssd/yangyuqiang/miniforge3/envs/isaac-sim42/lib/python3.10/site-packages/bpy/4.0/datafiles/studiolights/world/forest.exr", strength=1.0)
# 设置相机位置和方向
cam_pose = bproc.math.build_transformation_mat([0.0, 0.0, 0.0], [0.0, 0, 0.0])
for _ in range(2):
    bproc.camera.add_camera_pose(cam_pose)

# 设置光照
# light = bproc.types.Light()
# light.set_type("POINT")  # 点光源
# light.set_location([2, -2, 5])
# light.set_energy(500)  # 光源强度

# 设置渲染分辨率
# bpy.context.scene.render.resolution_x = 640
# bpy.context.scene.render.resolution_y = 480

# 设置输出路径
# output_dir = "./output"
# bproc.renderer.set_output_format("PNG")  # 输出格式为 PNG
import cv2
# 渲染场景
data = bproc.renderer.render()
cv2.imwrite("tt.jpg",data["colors"][1])
image_rgb = data["colors"][1]

cv2.imwrite("original_image.jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

# 反转 RGB 通道（RGB -> BGR）
image_bgr = image_rgb[:, :, ::-1]  # 使用切片操作反转通道顺序

# 保存通道反转后的图像
cv2.imwrite("reversed_channels_image.jpg", image_bgr)
print(data["colors"][1])
# bpy.ops.render.render()
# 打印渲染结果信息
# print("渲染完成！输出路径：", data)