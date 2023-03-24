# A simple script that uses blender to render views of a single object by sampling camera origin locations.
# Also produces depth map at the same time.
#
# Tested with Blender 2.9
#
# Example:
#  /usr/stud/zhouzh/blender-2.93.9-linux-x64/blender --background --python render_blender.py -- --output_folder ./tmp
# Batch rendering:
# find /storage/user/yenamand/ShapeNetCore.v1/02958343/ -name *.obj | head -3| xargs -n1 -P4 -I {} /usr/stud/zhouzh/blender-2.93.9-linux-x64/blender --background --python /usr/stud/zhouzh/pi-gan/blender_renderer/render_blender.py -- --output_folder /usr/stud/zhouzh/data/my_shapenetcar --views 10 {}
#
# Result directory:
# car_name
# |__./depth/
# |__./image/
# |__./normal/
# |__./img_shaded/
# |__./cameras.npz
# |__./pointcloud.npz

import argparse, sys, os, math, re
import bpy
import numpy as np
from utils import get_3x4_P_matrix_from_blender, sample_cam
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Renders given obj file by rotation a camera around it."
)
parser.add_argument(
    "obj",
    type=str,
    # default="/storage/user/yenamand/ShapeNetCore.v1/02958343/baa1e44e6e086c233e320a6703976361/model.obj",
    help="Path to the obj file to be rendered.",
)
parser.add_argument(
    "--views", type=int, default=4, help="number of views to be rendered"
)
parser.add_argument(
    "--output_folder",
    type=str,
    default="/tmp",
    help="The path the output will be dumped to.",
)
parser.add_argument(
    "--scale",
    type=bool,
    default=True,
    help="If apply scaling to the object.",
)
parser.add_argument(
    "--remove_doubles",
    type=bool,
    default=True,
    help="Remove double vertices to improve mesh quality.",
)
parser.add_argument(
    "--edge_split", type=bool, default=True, help="Adds edge split filter."
)
parser.add_argument(
    "--depth_scale",
    type=float,
    default=1,
    help="Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.",
)
parser.add_argument(
    "--color_depth",
    type=str,
    default="16",
    help="Number of bit per channel used for output. Either 8 or 16.",
)
parser.add_argument(
    "--format",
    type=str,
    default="PNG",
    help="Format of files generated. Either PNG or OPEN_EXR",
)
parser.add_argument(
    "--resolution", type=int, default=256, help="Resolution of the images."
)
parser.add_argument(
    "--engine",
    type=str,
    default="BLENDER_EEVEE",
    help="Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...",
)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print(Path(args.obj).stem)
# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine
render.image_settings.color_mode = "RGBA"  # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = args.color_depth  # ('8', '16')
render.image_settings.file_format = args.format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
render.film_transparent = True

scene.use_nodes = True
scene.view_layers["View Layer"].use_pass_normal = True
scene.view_layers["View Layer"].use_pass_diffuse_color = True
scene.view_layers["View Layer"].use_pass_object_index = True

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new("CompositorNodeRLayers")

# Create depth output nodes
depth_file_output = nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = "Depth Output"
depth_file_output.base_path = ""
depth_file_output.file_slots[0].use_node_format = True
depth_file_output.format.file_format = "OPEN_EXR"
depth_file_output.format.color_depth = args.color_depth
links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])
scale_node = nodes.new(type="CompositorNodeMixRGB")
scale_node.blend_type = "MULTIPLY"

# scale_node.use_alpha = True
scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs["Normal"], scale_node.inputs[1])

bias_node = nodes.new(type="CompositorNodeMixRGB")
bias_node.blend_type = "ADD"
# bias_node.use_alpha = True
bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_node.outputs[0], bias_node.inputs[1])

normal_file_output = nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = "Normal Output"
normal_file_output.base_path = ""
normal_file_output.file_slots[0].use_node_format = True
normal_file_output.format.file_format = args.format
links.new(bias_node.outputs[0], normal_file_output.inputs[0])

# Create albedo output nodes
alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
links.new(render_layers.outputs["DiffCol"], alpha_albedo.inputs["Image"])
links.new(render_layers.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = "Albedo Output"
albedo_file_output.base_path = ""
albedo_file_output.file_slots[0].use_node_format = True
albedo_file_output.format.file_format = args.format
albedo_file_output.format.color_mode = "RGBA"
albedo_file_output.format.color_depth = args.color_depth
links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])

# Create id map output nodes
id_file_output = nodes.new(type="CompositorNodeOutputFile")
id_file_output.label = "ID Output"
id_file_output.base_path = ""
id_file_output.file_slots[0].use_node_format = True
id_file_output.format.file_format = args.format
id_file_output.format.color_depth = args.color_depth

if args.format == "OPEN_EXR":
    links.new(render_layers.outputs["IndexOB"], id_file_output.inputs[0])
else:
    id_file_output.format.color_mode = "BW"

    divide_node = nodes.new(type="CompositorNodeMath")
    divide_node.operation = "DIVIDE"
    divide_node.use_clamp = False
    divide_node.inputs[1].default_value = 2 ** int(args.color_depth)

    links.new(render_layers.outputs["IndexOB"], divide_node.inputs[0])
    links.new(divide_node.outputs[0], id_file_output.inputs[0])

# Delete default cube
context.active_object.select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action="DESELECT")

bpy.ops.import_scene.obj(filepath=args.obj)

obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj

# Possibly disable specular shading
for slot in obj.material_slots:
    node = slot.material.node_tree.nodes["Principled BSDF"]
    node.inputs["Specular"].default_value = 0.05

points = np.asarray(
    [
        (bpy.data.objects["model"].matrix_world @ v.co)
        for v in bpy.data.objects["model"].data.vertices
    ]
)
breakpoint()
scale = 0.5 / max(-points.min(), points.max()) if args.scale else 1
loc = np.asarray(bpy.data.objects["model"].location)

if args.scale:
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.object.transform_apply(scale=True)
if args.remove_doubles:
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set(mode="OBJECT")
if args.edge_split:
    bpy.ops.object.modifier_add(type="EDGE_SPLIT")
    context.object.modifiers["EdgeSplit"].split_angle = 1.32645
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

# Note that all the rendered images, depth map, pcl etc are scaled, i.e the pcl is within the [-0.5, 0.5]^3 bbx.
points = np.asarray(
    [
        (bpy.data.objects["model"].matrix_world @ v.co)
        for v in bpy.data.objects["model"].data.vertices
    ]
)
pointcloud = {"points": points, "scale": 1 / scale, "loc": loc}

# Set objekt IDs
obj.pass_index = 1
# Make light just directional, disable shadows.
light = bpy.data.lights["Light"]
light.type = "SUN"
light.use_shadow = False
# Possibly disable specular shading:
light.specular_factor = 1.0
light.energy = 10.0

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.light_add(type="SUN")
light2 = bpy.data.lights["Sun"]
light2.use_shadow = False
light2.specular_factor = 1.0
light2.energy = 0.015
bpy.data.objects["Sun"].rotation_euler = bpy.data.objects["Light"].rotation_euler
bpy.data.objects["Sun"].rotation_euler[0] += 180

# Sample camera locations
cam_locations = sample_cam(args.views)

cam = scene.objects["Camera"]
cam.data.lens = 35
cam.data.sensor_width = 32
cam.data.sensor_height = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty

model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
fp = os.path.join(os.path.abspath(args.output_folder), model_identifier)

cameras = {}
for i in range(args.views):
    print(f"View {i}:")
    cam.location = cam_locations[i]
    render_file_path = os.path.join(fp, f"{i:04d}")
    scene.render.filepath = render_file_path
    depth_file_output.file_slots[0].path = os.path.join(fp, "depth", f"{i:04d}")
    normal_file_output.file_slots[0].path = os.path.join(fp, "normal", f"{i:04d}")
    albedo_file_output.file_slots[0].path = os.path.join(fp, "image", f"{i:04d}")
    id_file_output.file_slots[0].path = os.path.join(fp, "id", f"{i:04d}")
    bpy.ops.render.render(write_still=True)  # render still
    _, K, w2c = get_3x4_P_matrix_from_blender(bpy.data.objects["Camera"])
    cameras[f"world_mat_{i}"] = w2c
    cameras[f"world_mat_inv_{i}"] = np.linalg.inv(w2c)
    cameras[f"camera_mat_{i}"] = K
    cameras[f"camera_mat_inv_{i}"] = np.linalg.inv(K)
    # c2w = np.linalg.inv(w2c)
    # c2w[:, 1] *= -1
    # c2w[:, 2] *= -1
    # assert np.allclose(cameras[f"world_mat_inv_{i}"][:3, -1], cam.location)
    # assert np.allclose(np.asarray(cam.matrix_world), c2w, atol=0.01, rtol=0)

# Save the cameras and pointcloud
np.savez(os.path.join(fp, "cameras.npz"), **cameras)
np.savez(os.path.join(fp, "pointcloud.npz"), **pointcloud)

# Change the name of the rgb image
for p in (Path(fp) / "image").iterdir():
    des = os.path.join(fp, "image", p.stem[:4] + ".png")
    cmd = f"mv {str(p)} {des}"
    print(cmd)
    os.system(cmd)

img_shaded = Path(fp) / "img_shaded"
img_shaded.mkdir(exist_ok=True)
for p in Path(fp).iterdir():
    if p.suffix == ".png":
        cmd = f"mv {str(p)} {str(img_shaded)}"
        print(cmd)
        os.system(cmd)
# For debugging the workflow
# bpy.ops.wm.save_as_mainfile(filepath='debug.blend')
