import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import save_image
import im2mesh.common as common
import trimesh
import os
import torch
try:
    import open3d as o3d
except ImportError:
    pass

def visualize_data(data, data_type, out_file):
    r''' Visualizes the data with regard to its type.

    Args:
        data (tensor): batch of data
        data_type (string): data type (img, voxels or pointcloud)
        out_file (string): output file
    '''
    if data_type == 'img':
        if data.dim() == 3:
            data = data.unsqueeze(0)
        save_image(data, out_file, nrow=4)
    elif data_type == 'voxels':
        visualize_voxels(data, out_file=out_file)
    elif data_type == 'pointcloud':
        visualize_pointcloud(data, out_file=out_file)
    elif data_type is None or data_type == 'idx':
        pass
    else:
        raise ValueError('Invalid data_type "%s"' % data_type)


def visualize_voxels(voxels, out_file=None, show=False):
    r''' Visualizes voxel data.

    Args:
        voxels (tensor): voxel data
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    import ipdb; ipdb.set_trace()
    # Use numpy
    voxels = np.asarray(voxels)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    voxels = voxels.transpose(2, 0, 1)
    ax.voxels(voxels, edgecolor='k')
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)



def voxelgrid_to_trianglemesh(voxel, thresh=.5, mode='exact', normalize=True):
    """Converts passed voxel to a mesh
    Modified from Kaolin.

    Args:
        voxel (torch.Tensor): batched voxel array (num_shapes, d1, d2, d3)
        thresh (float): threshold from which to make voxel binary
        mode (str):
            -'exact': exect mesh conversion
            -'marching_cubes': marching cubes is applied to passed voxel
        normalize (bool): whether to scale the array to (-.5, .5)
    Returns:
        (verts_list (torch.Tensor), faces_list (torch.Tensor), bool):
            list of computed mesh properties (verts and faces), one element for each shape
            (bool): Indicates whether all meshes are empty (useful downstream).
    Example:
        >>> voxel = torch.ones([5, 32, 32, 32])
        >>> verts_list, faces_list, _ = voxelgrid_to_trianglemesh(voxel)
        >>> len(verts_list)
        5
    """
    assert (mode in ['exact', 'marching_cubes'])
    # voxel = threshold(voxel, thresh=thresh, inplace=False)
    device = voxel.device
    voxel_np = np.array((voxel.cpu() > thresh)).astype(bool)

    verts_list = []
    faces_list = []

    all_empty = True # whether all meshes are empty
    for i in range(voxel_np.shape[0]):
        if np.max(voxel_np[i, :, :, :]) == 0:
            # create empty mesh
            verts = torch.rand((3, 3), dtype=torch.float32, device=device)
            faces = torch.tensor([], dtype=torch.int64, device=device)
            verts_list.append(verts)
            faces_list.append(faces)
            continue

        all_empty = False
        trimesh_voxel = trimesh.voxel.VoxelGrid(voxel_np[i])

        if mode == 'exact':
            trimesh_voxel = trimesh_voxel.as_boxes()
        elif mode == 'marching_cubes':
            trimesh_voxel = trimesh_voxel.marching_cubes

        verts = torch.tensor(trimesh_voxel.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(trimesh_voxel.faces, dtype=torch.long, device=device)
        shape = torch.tensor(np.array(voxel[i].shape), dtype=torch.float, device=device)
        if normalize:
            # import ipdb; ipdb.set_trace()
            verts += 0.5 # NEW; UNCONFIRMED!
            verts /= shape
            verts = verts - .5

        verts_list.append(verts)
        faces_list.append(faces)

    return verts_list, faces_list, all_empty



def visualize_voxels_new(voxel, name, save_path, thresh=0.5, mode='marching_cubes'):
    # create save path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Convert voxel to mesh
    if not torch.is_tensor(voxel):
        voxel = torch.Tensor(voxel)
    # swap axes to .obj's +Y up, right-hand convention with order X-Z-Y
    # voxel = voxel.permute(0, 3, 2, 1)
    # # TODO: figure out why all 3 axes are in opposite direction when saving
    # voxel = torch.flip(voxel, dims=(1, 2, 3))
    # Covnert to .obj's coordinate from default coordinate

    verts_list, faces_list, _ = voxelgrid_to_trianglemesh(voxel, thresh=thresh, mode=mode, normalize=True)

    for i, (verts, faces) in enumerate(zip(verts_list, faces_list)):
        # Translate the mesh such that its centered at the origin.
        verts_max = verts.max()
        verts_min = verts.min()
        verts_mid = (verts_max + verts_min) / 2.
        verts = verts - verts_mid

        # add index to file name if more than one shape
        if len(verts_list) > 1:
            outfile = os.path.join(save_path, '%s_%d.obj'%(name, i))
        else:
            outfile = os.path.join(save_path, '%s.obj'%name)

        # Save .obj file (taken from Kaolin code)
        with open(outfile, 'w') as f:
            # write vertices
            for vert in verts:
                f.write('v %f %f %f\n' % tuple(vert))
            # write faces
            for face in faces:
                f.write('f %d %d %d\n' % tuple(face + 1))



def visualize_pointcloud_new(pointcloud, name, save_path):
    """Save a single pointcloud as .ply file."""
    # Open 3D can only store pointcloud as .ply
    save_file_ply = os.path.join(save_path, "{}.ply".format(name))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.io.write_point_cloud(save_file_ply, pcd)

    # Convert .ply to .obj and delete temp .ply file
    # ply_to_obj(save_file_ply, save_path, name, remove_ply=False)



def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.scatter(points[:, 2], points[:, 0], points[:, 1])
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.1, color='k'
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)


def visualise_projection(
        self, points, world_mat, camera_mat, img, output_file='out.png'):
    r''' Visualizes the transformation and projection to image plane.

        The first points of the batch are transformed and projected to the
        respective image. After performing the relevant transformations, the
        visualization is saved in the provided output_file path.

    Arguments:
        points (tensor): batch of point cloud points
        world_mat (tensor): batch of matrices to rotate pc to camera-based
                coordinates
        camera_mat (tensor): batch of camera matrices to project to 2D image
                plane
        img (tensor): tensor of batch GT image files
        output_file (string): where the output should be saved
    '''
    points_transformed = common.transform_points(points, world_mat)
    points_img = common.project_to_camera(points_transformed, camera_mat)
    pimg2 = points_img[0].detach().cpu().numpy()
    image = img[0].cpu().numpy()
    plt.imshow(image.transpose(1, 2, 0))
    plt.plot(
        (pimg2[:, 0] + 1)*image.shape[1]/2,
        (pimg2[:, 1] + 1) * image.shape[2]/2, 'x')
    plt.savefig(output_file)
