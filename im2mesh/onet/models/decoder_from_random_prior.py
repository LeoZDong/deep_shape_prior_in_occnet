from im2mesh.onet.models import decoder

from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist
from torch.optim import Adam
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
import os


def generate_n_points(voxels, n, bounds):
    """
    Generate n random points within bounds, with voxels interpreted as pixels
    Args:
        voxels: The voxels
        n: How many points to generate
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max) interpretation of the bounds for the voxels

    Returns:
        (points, points_occ)
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    xs = torch.rand(n)
    ys = torch.rand(n)
    zs = torch.rand(n)
    points = torch.stack((xs * (x_max - x_min) + x_min,
                          ys * (y_max - y_min) + y_min,
                          zs * (z_max - z_min) + z_min), 1)
    x_dim, y_dim, z_dim = voxels.shape
    indices = (torch.floor(xs * x_dim).long(), torch.floor(ys * y_dim).long(), torch.floor(zs * z_dim).long())
    points_occ = voxels[indices] > 0.5
    return points, points_occ


class DecoderOnlyModule(nn.Module):
    def __init__(self, decoder, device=None):
        super().__init__()
        self.init_z = torch.empty(decoder.z_dim).to(device)
        self.decoder = decoder
        nn.init.normal_(self.init_z)  # Initial as N(0, 1)

    def forward(self, input):
        """

        Args:
            input: The input (a batch of 3d points)

        Returns:
            The output
        """
        # print("Shape", input.shape)
        return self.decoder(input, self.init_z)


class DecoderOnlyTrainer(BaseTrainer):
    def __init__(self, model, device=None, vis_dir=None):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters())
        self.vis_dir = vis_dir
        self.probs = nn.Sigmoid()
        self.threshold = 0.5

    def compute_loss(self, points, points_occ):
        logits = self.model(points)
        loss = F.binary_cross_entropy_with_logits(logits, points_occ, reduction='mean')
        return loss

    def train_step(self, voxels, points=None, points_occ=None, n_points=None):
        if points is None:
            if n_points is None:
                raise ValueError("Either n_points or points should be specified")
            # points, points_occ = generate_n_points(voxels, n_points, (-1, 1, -1, 1, -1, 1))
            points, points_occ = generate_n_points(voxels, n_points, (-0.55, 0.55, -0.55, 0.55, -0.55, 0.55))
            points = points.to(self.device)
            points_occ = points_occ.float().to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(points, points_occ)
        loss.backward()
        self.optimizer.step()
        return loss


    def eval_step(self, points_input, points_occ):
        with torch.no_grad():
            points = points_input.to(self.device)
            points_occ = points_occ.to(self.device)

            self.model.eval()
            eval_dict = {}

            # for cross entropy loss validation
            logits = self.model(points)
            eval_dict['cross_entropy'] = F.binary_cross_entropy_with_logits(logits, points_occ, reduction='mean')

            # for iou validation
            m = nn.Sigmoid()
            predicted_occ = m(logits)

            pred_occ_np = (predicted_occ >= self.threshold).cpu().numpy()
            orig_occ_np = (points_occ >= self.threshold).cpu().numpy()
            iou = compute_iou(pred_occ_np, orig_occ_np).mean()
            eval_dict['iou'] = iou

            return eval_dict


    def visualize_decoder(self, it, loss, sub_dir=0):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        shape = (128, 128, 128)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        # p = p.expand(batch_size, *p.size())

        with torch.no_grad():
            p_r = self.model(p)

        # import ipdb; ipdb.set_trace()
        occ_hat = self.probs(p_r).view(1, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu()

        # for i in trange(batch_size):
            # input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            # vis.visualize_data(
            #     inputs[i].cpu(), self.input_type, input_img_path)

        # import ipdb; ipdb.set_trace()
        vis_dir = os.path.join(self.vis_dir, '{:06}'.format(sub_dir))
        vis.visualize_voxels_new(
            voxels_out, 'it{:06d}_{:.3f}'.format(it, loss), vis_dir)
