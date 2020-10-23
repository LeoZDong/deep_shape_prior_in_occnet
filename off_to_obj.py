import pymesh
import os

path = "/viscam/u/leozdong/occnet/demo/generation/viz/"

input_name = "00_mesh.off"
input_file = os.path.join(path, input_name)
output_name = "00_mesh.obj"
output_file = os.path.join(path, output_name)

mesh = pymesh.load_mesh(input_file)
pymesh.save_mesh(output_file, mesh)
