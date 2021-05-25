from glob import glob
from math import pi
from laspy.file import File
from util import *

def get_n_random_trees(tree_glob = "train_trees/*.las", num_trees_db=0):
    """
    Returns n random trees from the given directory
    the format of the returned trees are loaded pyntcloud objects
    """
    files = glob(tree_glob)
    
    trees = []
    for file in files:
        tree = PyntCloud.from_file(file)
        trees.append(tree)
    return trees


trees = get_n_random_trees()
amount_points = []
diameters = []
heights = []

for tree in trees:
    amount_points.append(tree.points.shape[0])
    diameters.append((tree.points["x"].max()**2 + tree.points["y"].max()**2)**0.5)
    heights.append(tree.points["z"].max())


print("amount_points", sum(amount_points) / len(amount_points))
print("diameters", sum(diameters) / len(diameters))
print("heights", sum(heights) / len(heights))


amount = sum(amount_points) / len(amount_points)
diam = sum(diameters) / len(diameters)
h = sum(heights) / len(heights)



average_density = amount/((diam*0.5)**2 * h * 3.14)
print("average_density", average_density)



