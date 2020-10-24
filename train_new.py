import torch
from torch import optim
import argparse
from im2mesh.onet.models import decoder
from im2mesh.onet.models.decoder_from_random_prior import DecoderOnlyTrainer, DecoderOnlyModule
from im2mesh.data import VoxelsField

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
npoints = 1000
trainer = DecoderOnlyTrainer(model, device=device)

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

voxel_field = VoxelsField("model.binvox")
# TODO: figure out what the 0, 0 means; make this cleaner
voxel_data = voxel_field.load('./data/ShapeNet/02958343/7c13a71834d2b97687cc3b689b9b258d/', 0, 0)

it = 0

print_every = 1000
while True:
    it += 1
    loss = trainer.train_step(voxel_data, n_points=10000)

    # Print output
    if print_every > 0 and (it % print_every) == 0:
        print('it=%03d loss=%.4f'
              % (it, loss))


