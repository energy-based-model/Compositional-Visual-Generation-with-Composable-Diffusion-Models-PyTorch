import torch
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from PIL import Image
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud

import os
import argparse
import open3d as o3d
import open3d.visualization.rendering as rendering


# create folders
plt_plot_folder = './plt_results'
mesh_folder = './mesh_results'

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, help="use '|' as the delimiter to compose separate sentences.")
parser.add_argument("--scale", type=float, default=3.0)
args = parser.parse_args()

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
    guidance_scale=[args.scale, 0.0],
    model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
)

# Set a prompt to condition on.
prompts = [x.strip() for x in args.prompt.split("|")]
file_name = "_".join(prompts)

# Produce a sample from the model.
samples = None
for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=prompts))):
    samples = x

pc = sampler.output_to_point_clouds(samples)[0]
fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
fig.savefig(os.path.join(plt_plot_folder, f'{file_name}.png'))

print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()

print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))

mesh = marching_cubes_mesh(
    pc=pc,
    model=model,
    batch_size=4096,
    grid_size=128, # increase to 128 for resolution used in evals
    progress=True,
)

with open(os.path.join(mesh_folder, f'{file_name}_{args.scale}.ply'), 'wb') as f:
    mesh.write_ply(f)


# generate 360 video
def generate_video():
    render = rendering.OffscreenRenderer(640, 480)
    mesh = o3d.io.read_triangle_mesh(args.file)
    mesh.compute_vertex_normals()

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultLit'

    render.scene.camera.look_at([0, 0, 0], [1, 1, 1], [0, 0, 1])
    render.scene.add_geometry('mesh', mesh, mat)

    def update_geometry():
        render.scene.clear_geometry()
        render.scene.add_geometry('mesh', mesh, mat)

    def generate_images():
        index = 0
        for i in range(128):
            # Rotation
            R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi / 32))
            mesh.rotate(R, center=(0, 0, 0))
            # Update geometry
            update_geometry()
            img = render.render_to_image()
            o3d.io.write_image(folder_name + "/{:05d}.png".format(index), img, quality=100)
            index += 1
            time.sleep(0.05)

    generate_images()
    image_list = []
    for filename in sorted(glob.glob(f'{folder_name}/*.png')):  # assuming gif
        im = Image.open(filename)
        image_list.append(im)

    image_list[0].save(os.path.join(mesh_folder, f'{file_name}.gif'), save_all=True,
                       optimizer=False, duration=5, append_images=image_list[1:], loop=0)


# generate video
generate_video()
