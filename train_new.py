import torch
from torch import optim
import argparse
from im2mesh.onet.models import decoder
from im2mesh.onet.models.decoder_from_random_prior import DecoderOnlyTrainer, DecoderOnlyModule
from im2mesh.data import VoxelsField, PointsField, PointCloudField
import os
import numpy as np

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

# args = parser.parse_args()
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

# Model
model = DecoderOnlyModule(decoder.Decoder(c_dim=0), device=device)

# Intialize training
shape_id = '7c13a71834d2b97687cc3b689b9b258d'
npoints = 1000
vis_dir = os.path.join('./visualize', shape_id, 'iterations')
trainer = DecoderOnlyTrainer(model, device=device, vis_dir=vis_dir)

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

# Load data
data_dir = './data/ShapeNet/02958343'
voxel_field = VoxelsField("model.binvox")
# TODO: figure out what the 0, 0 means; make this cleaner
voxel_data = torch.FloatTensor(voxel_field.load(os.path.join(data_dir, shape_id), 0, 0))
print(voxel_data.shape)

# Visualize initial inputs
# import ipdb; ipdb.set_trace()
from im2mesh.utils import visualize
save_path = os.path.join('./visualize', shape_id)
visualize.visualize_voxels_new(voxel_data.unsqueeze(0), 'input_voxel', save_path, mode='exact')

# pointclouds
# points_field = PointsField('points.npz')
# points = points_field.load(os.path.join(data_dir, shape_id), 0, 0)
pointcloud_field = PointCloudField('pointcloud.npz')
pointcloud = pointcloud_field.load(os.path.join(data_dir, shape_id), 0, 0)[None]
# import ipdb; ipdb.set_trace ()

#
# points_file = np.load(os.path.join(data_dir, shape_id, 'points.npz'))
# points = points_file['points']
# points_occ = points_file['occupancies'] / points_file['occupancies'].max()
# occupied_points = points[points_occ > 0.5]
#
# pointcloud = np.load(os.path.join(data_dir, shape_id, 'pointcloud.npz'))['points']

visualize.visualize_pointcloud_new(pointcloud, 'pointcloud', save_path)
# visualize.visualize_pointcloud_new(pointcloud, 'pointcloud', save_path)


# Test points sampling function
import ipdb; ipdb.set_trace()
from im2mesh.onet.models.decoder_from_random_prior import generate_n_points
bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
test_points = generate_n_points(voxel_data, 100000, bounds)[0].cpu().numpy()
visualize.visualize_pointcloud_new(test_points, 'test', save_path)



# Configure training loop
it = 0
print_every = 10
vis_every = 5000
while True:
    it += 1
    loss = trainer.train_step(voxel_data, n_points=10000)

    # Print output
    if print_every > 0 and (it % print_every) == 0:
        print('it=%03d loss=%.4f'
              % (it, loss))

    if vis_every > 0 and (it == 1 or (it % vis_every) == 0):
        print('Visualizing')
        trainer.visualize_decoder(it, loss)
