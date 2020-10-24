import pymesh
import os

path = "/viscam/u/leozdong/occnet/demo/generation/vis"

for i in range(9):
    input_name = "0{}_mesh.off".format(i)
    input_file = os.path.join(path, input_name)
    output_name = "0{}_mesh.obj".formate(i)
    output_file = os.path.join(path, output_name)

    mesh = pymesh.load_mesh(input_file)
    pymesh.save_mesh(output_file, mesh)
