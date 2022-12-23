import os
import time
import torch
import glob
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import open3d.visualization.rendering as rendering

from PIL import Image
from tqdm.auto import tqdm
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud

parser = argparse.ArgumentParser()
parser.add_argument("--prompts", type=str, nargs='+')
parser.add_argument("--weights", type=float, nargs='+', default=3.0)
args = parser.parse_args()

# create folders
plt_plot_folder = './plt_results'
mesh_folder = './mesh_results'
viewpoint_folder = './frame_results'
video_folder = './video_results'

os.makedirs(plt_plot_folder, exist_ok=True)
os.makedirs(mesh_folder, exist_ok=True)
os.makedirs(viewpoint_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M-textvec'
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
    guidance_scale=[args.weights, 0.0],
    model_kwargs_key_filter=('texts', ''),  # Do not condition the upsampler at all
)

print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()

print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))


def generate_pcd(prompt_list):
    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=prompt_list))):
        samples = x
    return samples


def generate_fig(samples):
    pc = sampler.output_to_point_clouds(samples)[0]
    fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)))
    return fig, pc


def generate_mesh(pc):
    mesh = marching_cubes_mesh(
        pc=pc,
        model=model,
        batch_size=4096,
        grid_size=128,  # increase to 128 for resolution used in evals
        progress=True,
    )
    return mesh


# generate 360 video
def generate_video(mesh_path):
    render = rendering.OffscreenRenderer(640, 480)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultLit'

    render.scene.camera.look_at([0, 0, 0], [1, 1, 1], [0, 0, 1])
    render.scene.add_geometry('mesh', mesh, mat)

    def update_geometry():
        render.scene.clear_geometry()
        render.scene.add_geometry('mesh', mesh, mat)

    def generate_images():
        for i in range(64):
            # Rotation
            R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi / 32))
            mesh.rotate(R, center=(0, 0, 0))
            # Update geometry
            update_geometry()
            img = render.render_to_image()
            o3d.io.write_image(os.path.join(viewpoint_folder, "{:05d}.jpg".format(i)), img, quality=100)
            time.sleep(0.05)

    generate_images()
    image_list = []
    for filename in sorted(glob.glob(f'{viewpoint_folder}/*.jpg')):  # assuming gif
        im = Image.open(filename)
        image_list.append(im)
    return image_list


if __name__ == '__main__':
    # Set a prompt to condition on.
    file_name = "_".join(args.prompts)
    pcd = generate_pcd(args.prompts)

    # save fig visualization
    fig, pc = generate_fig(pcd)
    fig.savefig(os.path.join(plt_plot_folder, f'{file_name}.png'))

    # save mesh file
    mesh = generate_mesh(pc)
    mesh_path = os.path.join(mesh_folder, f'{file_name}_{args.weights}.ply')
    with open(mesh_path, 'wb') as f:
        mesh.write_ply(f)

    # generate video
    image_frames = generate_video(mesh_path)
    gif_path = os.path.join(video_folder, f'{file_name}.gif')
    image_frames[0].save(gif_path, save_all=True, optimizer=False, duration=5, append_images=image_frames[1:], loop=0)
