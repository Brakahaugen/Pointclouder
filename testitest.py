
from glob import glob
from random import shuffle
from pyntcloud import PyntCloud
from laspy.file import File
import numpy as np


num_trees = 10
tree_glob = "train_trees/*.las"

files = glob(tree_glob)
shuffle(files)
files = files[:num_trees] if len(files) >= num_trees else files

p = PyntCloud.from_file("train_trees/Tree3.las")
print(p.points)
# trees = []
# for file in files:
#     trees.append(PyntCloud.from_file(file))
#     print(trees[-1].points)
#     print()
#     print(File(file, mode='r').get_x())
#     print("\n\n")