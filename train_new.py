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
# Used for visualization only
pointcloud_field = PointCloudField('pointcloud.npz')
pointcloud = pointcloud_field.load(os.path.join(data_dir, shape_id), 0, 0)[None]

# Points are points randomly sampled in space with an associated occupancy
# Used for validation
points_field = PointsField('points.npz', unpackbits=True)
points_dict = points_field.load(os.path.join(data_dir, shape_id), 0, 0)
points = torch.FloatTensor(points_dict[None])
points_occ = torch.FloatTensor(points_dict['occ'])


#### Visualize initial inputs ####
from im2mesh.utils import visualize
save_path = os.path.join('./visualize', shape_id)
visualize.visualize_voxels_new(voxel_data.unsqueeze(0), '1_input_voxel_exact', save_path, mode='exact')
visualize.visualize_voxels_new(voxel_data.unsqueeze(0), '1_input_voxel_mc', save_path, mode='marching_cubes')
# points sampling function
from im2mesh.onet.models.decoder_from_random_prior import generate_n_points
bounds = (-0.55, 0.55, -0.55, 0.55, -0.55, 0.55)
test_points, points_occ_gen = generate_n_points(voxel_data, 100000, bounds)
test_points = test_points.cpu().numpy()[points_occ_gen > 0.5]
visualize.visualize_pointcloud_new(test_points, '2_input_points', save_path)
visualize.visualize_pointcloud_new(pointcloud, '2_surface_points', save_path)
visualize.visualize_pointcloud_new(points[points_occ > 0.5].cpu().numpy(), '2_eval_points', save_path)



def plot_metric(records, its, plot_title, filename, start_it=1):
    import matplotlib.pyplot as plt
    import pandas as pd

    if len(its) > 0 and its[-1] > start_it:
        x = np.array(its)
        y = np.array(records)

        y = pd.DataFrame(y).rolling(window=10).mean().to_numpy()
        # y = pd.rolling_mean(y, window=10)

        fig, ax = plt.subplots()
        ax.plot(x[np.where(x >= start_it)], y[np.where(x >= start_it)])
        ax.set(xlabel='iteration', ylabel='metric',
               title=plot_title)

        fig.savefig(os.path.join(save_path, filename), dpi=1000)
        plt.close()


def plot_smooth_eval(entropy_rec, iou_rec, window_size):
    '''
    Plots the moving average of 100 epochs for the two evaluation metrics
    Assume that at least 100 iteration have been performed

    Args:
        entropy_rec: list of cross entropy loss calculations at each iteration
        iou_rec: list of iou calculations at each iteration
        window_size: number of iterations used for moving average
    '''
    import pandas as pd
    import numpy as np

    import ipdb; ipdb.set_trace()
    # x = np.arange(1, len(entropy_rec) + 1, 1)
    df_entropy = pd.DataFrame(entropy_rec)
    df_iou = pd.DataFrame(iou_rec)

    entropy_sma = df_entropy.rolling(window_size).mean()
    iou_sma = df_iou.rolling(window_size).mean()

    ax1 = entropy_sma.plot(title='Moving Average for Cross Entropy Loss')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    fig_entropy = ax1.get_figure()

    ax2 = iou_sma.plot(title='Moving Average for IoU')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("IoU")
    fig_iou = ax2.get_figure()

    save_path = "."
    fig_entropy.savefig(os.path.join(save_path, "smootheval_entropy.png"), dpi=1000)
    fig_iou.savefig(os.path.join(save_path, "smootheval_iou.png"), dpi=1000)

# Configure training loop
it = 0
print_every = 10
vis_every = 1000
plot_every = 1000
eval_every = 10
# metrics records
loss_rec = []
entropy_rec = []
iou_rec = []
eval_it = []
while True:
    it += 1
    # Train step
    loss = trainer.train_step(voxel_data, n_points=10000)
    loss_rec.append(loss.item())

    # Evaluation step
    if eval_every > 0 and (it % eval_every) == 0:
        eval_dict = trainer.eval_step(points, points_occ)
        entropy_eval = eval_dict['cross_entropy']
        iou_eval = eval_dict['iou']
        entropy_rec.append(entropy_eval.item())
        iou_rec.append(iou_eval.item())
        eval_it.append(it)

    # Print output
    if print_every > 0 and (it % print_every) == 0:
        print('it=%03d loss=%.4f entropy_eval=%0.4f iou_eval=%0.4f)'
              % (it, loss, eval_dict['cross_entropy'], eval_dict['iou']))

    # Visualize shape
    if vis_every > 0 and (it == 1 or (it % vis_every) == 0):
        print('Visualizing...')
        sub_dir = (it // 10000) * 10000
        trainer.visualize_decoder(it, loss, sub_dir)

    # Plot metrics
    if plot_every > 0 and (it == 1 or (it % plot_every) == 0):
        print("Plotting...")
        plot_metric(loss_rec, np.arange(1, len(loss_rec) + 1, 1), "Training loss starting at iteration 500", '0_loss.png', start_it=500)
        plot_metric(entropy_rec, eval_it, "Validation cross entropy", '0_entropy.png', start_it=1)
        plot_metric(iou_rec, eval_it, "Validation IoU", '0_iou.png', start_it=1)
