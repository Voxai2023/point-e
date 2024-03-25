from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter
from PIL import Image
import torch
from tqdm.auto import tqdm
import open3d as o3d
import numpy as np
import math
import gradio as gr
from fastapi import FastAPI

# setting up point-e
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initial_point_e():
    print('creating base model...')
    base_name = 'base1B'  # use base300M(1.25G) or base1B(5G) for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )

    return sampler


def create_point_cloud(img):
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
        samples = x
    pc = sampler.output_to_point_clouds(samples)[0]
    return pc


def voxelization(pc):
    # Load point cloud and create pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.coords)
    pcd.colors = o3d.utility.Vector3dVector(np.vstack((pc.channels["R"], pc.channels["G"], pc.channels["B"])).transpose())
    # o3d.visualization.draw_geometries([pcd])

    # Voxelization
    vsize = max(pcd.get_max_bound() - pcd.get_min_bound()) * 0.05
    vsize = round(vsize, 4)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=vsize)
    return voxel_grid, vsize


def color_distance(c1, c2):
    (r1,g1,b1) = c1.r, c1.g, c1.b
    (r2,g2,b2) = c2.r, c2.g, c2.b
    return math.sqrt((r1 - r2)**2 + (g1 - g2) ** 2 + (b1 - b2) **2)


def nearest_color(color, palette):
    colors_dict = {}
    for i in range(len(palette)):
        colors_dict[i] = palette[i]
    closest_colors = sorted(colors_dict, key=lambda point: color_distance(color, colors_dict[point]))
    return colors_dict[closest_colors[0]]


def nearest_color_index(color, palette):
    color = nearest_color(color, palette)
    return palette.index(color)


def try_add_color_to_palette(new_color, palette, color_threshold=24):
    if len(palette) >= 254:
        return palette, nearest_color_index(new_color, palette)
    for color in palette:
        if color_distance(new_color, color) <= color_threshold:
            return palette, nearest_color_index(new_color, palette)
    palette.append(new_color)
    return palette, (len(palette)-1)


def write_vox_file(voxel_grid, output_name):
    voxels = voxel_grid.get_voxels()  # returns list of voxels
    indices = np.stack(list(vx.grid_index for vx in voxels))

    # size of whole model
    maxIndex = 0
    for i in indices:
        max_number = max(i)
        if max_number > maxIndex:
            maxIndex = max_number
    maxIndex = maxIndex + 1

    # new color indexes match with rgba_palette
    color_index = []

    # rgba color code formatting
    palette = []

    a = np.zeros((maxIndex, maxIndex, maxIndex), dtype=int)

    for v in voxels:
        coord = v.grid_index
        color = v.color
        color = Color(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255)
        threshold = max(7, min(12, len(palette) * 0.65))
        palette, color_index = try_add_color_to_palette(color, palette, color_threshold=threshold)
        a[coord[1], (maxIndex - 1) - coord[2], coord[0]] = color_index + 1

    vox = Vox.from_dense(a)
    vox.palette = palette
    VoxWriter('output/' + output_name + '.vox', vox).write()


def write_display_file(voxel_grid, output_name, vsize):
    # Make it a mesh
    voxels = voxel_grid.get_voxels()
    vox_mesh = o3d.geometry.TriangleMesh()
    for v in voxels:
        cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube.paint_uniform_color(v.color)
        cube.translate(v.grid_index, relative=False)
        vox_mesh += cube

    vox_mesh.translate([0.5, 0.5, 0.5], relative=True)
    vox_mesh.scale(vsize, [0, 0, 0])

    vox_mesh.merge_close_vertices(0.0000001)
    T = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    vox_mesh.translate(voxel_grid.origin, relative=True)

    o3d.io.write_triangle_mesh("output/" + output_name + '.gltf', vox_mesh.transform(T))


def create_model(image, output_name, markdown):
    pc = create_point_cloud(image)
    voxel_grid, vsize = voxelization(pc)
    write_display_file(voxel_grid, output_name, vsize)
    write_vox_file(voxel_grid, output_name)
    return ['output/' + output_name + '.gltf', 'output/' + output_name + '.vox']
    # return ['output/testing1.gltf', 'output/testing1.vox']


# global sampler
sampler = initial_point_e()
app = FastAPI()

# img = Image.open('data/coconut_tree.png')
# create_model(img, 'test_object')

demo = gr.Interface(
    fn=create_model,
    inputs=[
        gr.Image(type='pil'),
        gr.Text(label='Output file name without file format'),
        gr.Markdown(
        """
        # Steps:
        1. Select a image
        2. Input file name
        3. Click submit button
        
        *Average model processing time is 6 minutes.
        
        *You will need to wait longer if there are other user generating models.
        """, line_breaks=True)
    ],
    outputs=[
        gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="gltf model", camera_position=[50, None, None]),
        gr.File(type='filepath',  label="Vox model")
    ],
    allow_flagging="never"
)

demo.queue(max_size=5)


@app.get("/")
def read_main():
    return {"message": "This is your main app"}


app = gr.mount_gradio_app(app, demo, path='/demo')

