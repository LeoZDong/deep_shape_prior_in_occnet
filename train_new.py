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

# Define model
model = DecoderOnlyModule(decoder.Decoder(c_dim=0), device=device)

#### Intialize training ####
shape_id = '7c13a71834d2b97687cc3b689b9b258d'
# shape_id = '24d07a3c5ff0840872152988eac576ab'
# shape_id = '36190ce6fe041e452d647b1c17442c93'  # does not have model
# shape_id = '49c2f144b928726572a38ac2b8f5cd48'
# shape_id = '53737a4b45fc06963ffe0e5069bf1eb5'
vis_dir = os.path.join('./visualize', shape_id, 'iterations')
os.system('rm -rf {}'.format(vis_dir))
trainer = DecoderOnlyTrainer(model, device=device, vis_dir=vis_dir)

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)


#### Load data ####
# Training data
data_dir = './data/ShapeNet/02958343'
voxel_field = VoxelsField("model.binvox")
# TODO: figure out what the 0, 0 means; make this cleaner
voxel_data = torch.FloatTensor(voxel_field.load(os.path.join(data_dir, shape_id), 0, 0))
print(voxel_data.shape)

# Pointclouds are points sampled from the surface of the mesh
pointcloud_field = PointCloudField('pointcloud.npz')
pointcloud = pointcloud_field.load(os.path.join(data_dir, shape_id), 0, 0)[None]

# Points are points randomly sampled in space with an associated occupancy
points_field = PointsField('points.npz', unpackbits=True)
points_dict = points_field.load(os.path.join(data_dir, shape_id), 0, 0)
points = points_dict[None]
points_occ = points_dict['occ']


#### Visualize initial inputs ####
from im2mesh.utils import visualize
save_path = os.path.join('./visualize', shape_id)
visualize.visualize_voxels_new(voxel_data.unsqueeze(0), 'input_voxel_exact', save_path, mode='exact')
visualize.visualize_voxels_new(voxel_data.unsqueeze(0), 'input_voxel_mc', save_path, mode='marching_cubes')
# points sampling function
from im2mesh.onet.models.decoder_from_random_prior import generate_n_points
bounds = (-0.55, 0.55, -0.55, 0.55, -0.55, 0.55)
test_points, points_occ_gen = generate_n_points(voxel_data, 100000, bounds)
test_points = test_points.cpu().numpy()[points_occ_gen > 0.5]
visualize.visualize_pointcloud_new(test_points, 'test', save_path)
visualize.visualize_pointcloud_new(pointcloud, 'pointcloud', save_path)
# visualize.visualize_pointcloud_new(pointcloud, 'points', save_path)



#

def plot_loss(loss_rec):
    import matplotlib
    import matplotlib.pyplot as plt
    start = 500

    if len(loss_rec) > start:
        x = np.arange(1, len(loss_rec) + 1, 1)
        y = loss_rec
        fig, ax = plt.subplots()
        ax.plot(x[start:], y[start:])
        ax.set(xlabel='iteration', ylabel='loss',
               title='Loss record starting at 500 iterations')

        fig.savefig(os.path.join(save_path, "loss.png"), dpi=1000)

def plot_eval(entropy_rec, iou_rec):
    import matplotlib
    import matplotlib.pyplot as plt

    x = np.arange(1, len(entropy_rec) + 1, 1)
    y1 = entropy_rec
    y2 = iou_rec

    fig_entropy, ax1 = plt.subplots()
    fig_iou, ax2 = plt.subplots()

    ax1.plot(x, y1)
    ax2.plot(x, y2)
    ax1.set(xlabel='iteration', ylabel='eval',
           title='entropy record starting at 0')
    ax2.set(xlabel='iteration', ylabel='eval',
           title='iou record starting at 0')

    fig_entropy.savefig(os.path.join(save_path, "eval_entropy.png"), dpi=1000)
    fig_iou.savefig(os.path.join(save_path, "eval_iou.png"), dpi=1000)


# Configure training loop
it = 0
print_every = 10
vis_every = 1000
plot_every = 5000
loss_rec = []
## for eval
entropy_rec = []
iou_rec = []
while True:
    it += 1
    loss = trainer.train_step(voxel_data, n_points=10000)
    loss_rec.append(loss.item())

    ## for eval
    eval_dict = trainer.eval_step(points, points_occ)
    entropy_eval = eval_dict['cross_entropy']
    iou_eval = eval_dict['iou']
    entropy_rec.append(entropy_eval.item())
    iou_rec.append(iou_eval.item())

    # Print output
    if print_every > 0 and (it % print_every) == 0:
        print('it=%03d loss=%.4f entropy_eval=%0.4f iou_eval=%0.4f)'
              % (it, loss, eval_dict['cross_entropy'], eval_dict['iou']))

    # Visualize shape
    if vis_every > 0 and (it == 1 or (it % vis_every) == 0):
        print('Visualizing...')
        sub_dir = (it // 10000) * 10000
        trainer.visualize_decoder(it, loss, sub_dir)

    # Plot loss
    if plot_every > 0 and (it == 1 or (it % plot_every) == 0):
        print("Plotting loss...")
        plot_loss(loss_rec)

        ## for eval
        print("Plotting eval...")
        plot_eval(entropy_rec, iou_rec)
