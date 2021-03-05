from util import *
import pandas as pd
from glob import glob
from random import random, shuffle, uniform, randint




def get_n_random_trees(num_trees, tree_glob = "test_trees/*.las"):
    """
    Returns n random trees from the given directory
    the format of the returned trees are loaded pyntcloud objects
    """
    files = glob(tree_glob)
    shuffle(files)
    files = files[:num_trees] if len(files) >= num_trees else files

    trees = []
    for file in files:
        trees.append(PyntCloud.from_file(file))

    return trees


def put_in_origo(pc: pd.DataFrame, cols: list):
    """
    Sets the minimum point of the dataframe into origo, and translates all other points accordingly
    Returns: Dataframe with points in range [0 -> (xmax-xmin)]
    """
    pc[cols] = pc[cols].apply(lambda x: (x - x.min()), axis=0)
    return pc

def get_maximum_xyz(pcs: list, cols: list):
    """
    returns the maximum x, y and z value from the given pcs (pointclouds)
    """
    max_vals = [0,0,0]

    for tree in pcs:
        tree = tree.points
        for i in range(len(cols)):
            if tree[cols[i]].max() > max_vals[i]:
                max_vals[i] = tree[cols[i]].max()

    return max_vals

def normalize_trees_on_max_xyz(random_trees: list, cols: list, col_max_values: list):
    """
    For each tree in the random trees normalize the cols on the col max values
    Returns -> list of trees in range [0->1]
    """
    for tree in random_trees:
        for col, col_max in zip(cols, col_max_values):
            tree.points[col] = tree.points[col].div(col_max)
    return random_trees

def preprocess_trees(random_trees: list):
    """
    For each tree, put the lowest point in origo
    For all trees, find maximum value, divide all trees on this maxima.
    For all trees, scatter 
    """
    cols = ['x','y','z']

    for tree in random_trees:
        tree.points = put_in_origo(tree.points, cols)
    col_max_values = get_maximum_xyz(random_trees, cols)
    random_trees = normalize_trees_on_max_xyz(random_trees, cols, col_max_values)

        
    return random_trees

def create_random_space(width, height, num_trees):
    """
    Returns a list with size num_trees of random, unique integer pairs
    """
    width = width - 1
    height = height -1

    assert(num_trees < width * height), "Not big enough grid to create that many separated trees"

    random_pairs = []
    x, y = randint(0, width), randint(0, height)    

    while len(random_pairs) < num_trees:
        random_pairs.append((x, y))
        x, y = randint(0, width), randint(0, height)
        while (x, y) in random_pairs:
            x, y = randint(0, width), randint(0, height)

    return random_pairs
        
        
def scatter_trees_on_grid(random_trees: list, width_height: list, allow_clustering: bool, augmentation: dict):
    """
        Scatters the trees randomly on a grid
        Returns a PyntCloud with all the trees scattered and merged into one
    """
    axes = ["x","y"]
    label_id = 1

    if allow_clustering:
        for tree in random_trees:
            for axis, max in zip(axes, width_height):
                tree.points[axis] = tree.points[axis].add(round(uniform(0,max-1), 3))
                tree.points["label_id"] = label_id
            label_id += 1

    if not allow_clustering:
        for tree, offsets in zip(random_trees, create_random_space(*width_height, len(random_trees))):
            for axis, val in zip(axes, offsets):
                tree.points[axis] = tree.points[axis].add(val)
                tree.points["label_id"] = label_id
            label_id += 1


    # print("single tree", random_trees[0].points)
    merged_trees = random_trees[0]
    for tree in random_trees[1:]:
        merged_trees.points = pd.concat([merged_trees.points, tree.points])

    print(merged_trees.points)

    return merged_trees

# my1 = PyntCloud.from_file("test_trees/1.las")
# my2 = PyntCloud.from_file("test_trees/2.las")
# ts = preprocess_trees([my1,my2])
# # print(ts[0].points, ts[1].points)



def get_random_sample(num_trees = 16, width: int = 8, height: int = 8, resolution: int = 512, allow_clustering: bool = True, augmentation: dict = {}):
    print("getting")
    random_trees = get_n_random_trees(num_trees)
    print("processing")
    random_trees = preprocess_trees(random_trees)
    print("scattering")
    scattered_trees_on_grid = scatter_trees_on_grid(random_trees, [width, height], allow_clustering=allow_clustering, augmentation=augmentation)

    print("normalizing")
    scattered_trees_on_grid.points = normalize(scattered_trees_on_grid.points)

    print("Creating sequences")
    sequences = TDI(scattered_trees_on_grid.points, 3)
    images = []
    print("Mapping")
    for seq in sequences:
        images.append(Mapping_M(seq, r = resolution))
    
    label_image = create_label_image(sequences[-1], r = resolution)

    print(images[0])
    print("concattenating images")
    concatted_image = np.append(np.append(images[0], images[1], axis=1), images[2], axis=1)

    cv2.imshow("windows", concatted_image)
    cv2.imshow("windowsss", label_image)
    cv2.waitKey(0)
    return concatted_image #A concatted image with the three images


if __name__ == "__main__":
    print("starting program")
    get_random_sample()