from util import *
import pandas as pd
from glob import glob
from random import shuffle, uniform, randint
import os.path
from laspy.file import File
from shapely.geometry import Polygon, Point



def get_n_random_trees(num_trees, tree_glob = "train_trees/*.las"):
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
    # return random_trees

    #This method is probably more correct, because the preserve x,y relations
    tree.points["x"] = tree.points["x"].div(max(col_max_values[1:]))
    tree.points["y"] = tree.points["y"].div(max(col_max_values[1:]))
    tree.points["z"] = tree.points["z"].div(col_max_values[2])
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


    merged_trees = random_trees[0]
    for tree in random_trees[1:]:
        merged_trees.points = pd.concat([merged_trees.points, tree.points])

    return merged_trees



def get_random_sample(num_trees = 32, width: int = 8, height: int = 8, resolution: int = 64, allow_clustering: bool = True, augmentation: dict = {}, tree_glob: str = "train_trees/*.las", image_id = 0):
    ("getting")
    random_trees = get_n_random_trees(num_trees, tree_glob=tree_glob)
    ("processing")
    random_trees = preprocess_trees(random_trees)
    ("scattering")
    scattered_trees_on_grid = scatter_trees_on_grid(random_trees, [width, height], allow_clustering=allow_clustering, augmentation=augmentation)

    ("normalizing")
    scattered_trees_on_grid.points = normalize(scattered_trees_on_grid.points)

    ("Creating sequences")
    sequences = TDI(scattered_trees_on_grid.points, 3)
    image = Mapping_M(sequences[-1], r = resolution)
    # images = []
    # ("Mapping")
    # for seq in sequences:
    #     images.append(Mapping_M(seq, r = resolution))
    # ("concattenating images")
    # concatted_image = np.append(np.append(images[0], images[1], axis=1), images[2], axis=1)
 
    label_image, label = create_label_image(sequences[-1], r = resolution, image_id=image_id)
    return image, label_image, label #A image
    

def create_test_image(resolution=64, dir_id: str = "sub1", image_id=0):
    
    merged_trees = PyntCloud.from_file("test_trees/subareas/" + dir_id + ".las")

    print("POINTS HALLO BROR:", merged_trees.points)

    merged_trees.points["label_id"] = merged_trees.points["original_cloud_index"] + 1
    print(merged_trees.points["label_id"].max())
    print(merged_trees.points["label_id"].min())
    

    merged_trees.points = normalize(merged_trees.points)
    # merged_trees.to_file("normal.xyz")


    ("Creating sequences")
    sequences = TDI(merged_trees.points, 3)
    image = Mapping_M(sequences[-1],  r = resolution)
    # images = []
    # ("Mapping")
    # for seq in sequences:
    #     images.append(Mapping_M(seq, r = resolution))
    # ("concattenating images")
    # concatted_image = np.append(np.append(images[0], images[1], axis=1), images[2], axis=1)
    
    label_image, label = create_label_image(sequences[-1], r = resolution, image_id=image_id)
    return image, label_image, label #A image
    
    
# if __name__ == "__main__":
# example, target = create_test_image()
# cv2.imwrite("data/test/images/"+str(0)+".png", example) 
# cv2.imwrite("data/test/labels/"+str(0)+".png", target)
# import sys
# sys.exit()

def create_manual_label_image(r, ls, size = 10, width = 2, image_id = 0):
    I = np.zeros((r,r))

    for l in ls:
        l  = l["segmentation"][0]
        l = [l[i:i+2] for i in range(0, len(l), 2)]        # Use xrange in py2k

        c = Polygon(l).centroid
        c = c.coords[0]

        for x in range(-size + 1, size):
            for y in range(-width + 1, width):
                try:
                    I[int(c[1] + x), int(c[0] + y)] = 255
                except:
                    ("whatever")

        for y in range(-size + 1, size):
            for x in range(-width + 1, width):
                try:
                    I[int(c[1] + x), int(c[0] + y)] = 255
                except:
                    ("whatever")

    cv2.imwrite("test_image.png", I)
    return 
            

if __name__ == "__main__":
    
    num_samples = 20000

    resolution = 128
    train_val_ratio = 5

    test_ims = ["sub1", "sub2", "sub3", "sub4"]
    for i in range(len(test_ims)):
        example, target, labels = create_test_image(resolution=resolution, dir_id=test_ims[i], image_id = i)
        cv2.imwrite("data/test/images/"+str(i)+".png", example) 

        create_manual_label_image(resolution, labels, image_id = i)

        with open("data/test/labels/"+str(i)+".json", "w") as f:
            f.write(json.dumps(labels, indent = 4))
            f.close()

    for i in tqdm(range(num_samples)):
        if os.path.exists("data/train/images/"+str(i)+".png"):
            continue

        width_height = randint(3,8)
        num_trees = randint(4,25)

        example, target, labels = get_random_sample(num_trees, tree_glob="train_trees/*.las", image_id=i, resolution=resolution,  width=width_height, height = width_height)
        cv2.imwrite("data/train/images/"+str(i)+".png", example) 
        with open("data/train/labels/"+str(i)+".json", "w") as f:
            f.write(json.dumps(labels, indent = 4))
            f.close()

        

        if i % train_val_ratio == 0:
            example, target, labels = get_random_sample(num_trees=num_trees, tree_glob="test_trees/*.las", image_id=int(i/train_val_ratio), resolution=resolution, width=width_height, height = width_height)
            cv2.imwrite("data/val/images/"+str(int(i/train_val_ratio))+".png", example) 
            with open("data/val/labels/"+str(int(i/train_val_ratio))+".json", "w") as f:
                f.write(json.dumps(labels, indent = 4))
                f.close()
