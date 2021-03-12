from pyntcloud import PyntCloud
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
import cv2
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import json


def fill_holes(I: np.array):
    new_im = []

    for row in (I):
        starts = {}
        ends = {}
        new_row = row.copy()
        for i in range(len(row)):
            if (row[i]) != 0 and (row[i]) not in starts:
                starts[(row[i])] = i
            elif (row[i]) != 0:
                ends[(row[i])] = i
                
        for key, val in ends.items():
            
            for i in range(starts[key], ends[key]):
                new_row[i] = key
        new_im.append(new_row)

    I = np.asarray(new_im)
    new_im = []
    
    for row in (I.T):
        starts = {}
        ends = {}
        new_row = row.copy()
        for i in range(len(row)):
            if (row[i]) != 0 and (row[i]) not in starts:
                starts[(row[i])] = i
            elif (row[i]) != 0:
                ends[(row[i])] = i
                
        for key, val in ends.items():

            for i in range(starts[key], ends[key]):
                new_row[i] = key

        new_im.append(new_row)

    I = np.asarray(new_im).T
    return I

def TDI(points: pd.DataFrame, k = 3):
    """
    Adds a column 'slice' to the dataframe. the k slices are equal in length
    given k=3:
        slice 1 is the scattered shrub
        slice 2 is the rod-shaped trunk
        slice 3 is the spherical canopy
    returns a list of the sequences 
    """

    points['slice'] = points.apply(lambda row: math.ceil(row["z"] * k) if math.ceil(row["z"] * k) > 0 else 1, axis=1)

    slices = []
    for s in range(1,k +1):
        slices.append(points[points['slice'] <= s])

    return slices

def create_sub_masks(mask_image):
    width, height = mask_image.shape

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image[x,y]

            # If the pixel is not black...
            if pixel != 0:
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = np.zeros((width+2,height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str][x+1,y+1] = 1

    return sub_masks


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id = 0):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)


    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    
    if not multi_poly.bounds:
        return None

    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': 0,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }
    annotation_id += 1
    return annotation

def create_labels(I, image_id):
    submasks = create_sub_masks(I)
    # print(submasks)
    
    annotations = []
    annotation_id = 0
    for color, sub_mask in submasks.items():
        category_id = 1
        annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id)
        if annotation:
            annotations.append(annotation)
            annotation_id += 1        

    return annotations
    

def create_label_image(slice: pd.DataFrame, r: int = 256, image_id = 0):

    n = slice.shape[0]
    x_list = list(slice['x'])
    y_list = list(slice['y'])
    label_list = list(slice["label_id"])
    
    I = np.zeros((r,r))
    for j in (range(n)):
        x = math.floor(x_list[j] * r) if math.floor(x_list[j] * r) < r else r-1
        y = math.floor(y_list[j] * r) if math.floor(y_list[j] * r) < r else r-1
        I[x,y] = label_list[j]
    
    I = fill_holes(I)
    
    label = create_labels(I, image_id)

    # SUBJECT TO CHANGE
    # for i in range(I.shape[0]):
    #     for j in range(I.shape[1]):
    #         if I[i,j] != 0:
    #             I[i,j] = 1
    
    return I, label
        

def Mapping_M(slice: pd.DataFrame, r: int = 256):
    n = slice.shape[0]

    x_list = list(slice['x'])
    y_list = list(slice['y'])
    
    I = np.zeros((r,r,3))
    C = {} #C = (r, r)
    for i in range(r):
        for j in range(r):
            C[i,j] = []

    for j in (range(n)):
        x = math.floor(x_list[j] * r) if math.floor(x_list[j] * r) < r else r-1
        y = math.floor(y_list[j] * r) if math.floor(y_list[j] * r) < r else r-1
        I[x,y,0] = I[x,y,0] + 1 #Appending to the density
        C[x,y].append(slice.iloc[j]['z'])

    for h in (range(r)):
        for t in range(r):
            z = C[h,t]
            val = abs(max(z) - min(z)) if z else 0
            I[h,t,1] = val

    for h in (range(r)):
        for t in range(r):
            grad_h = 0
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    if h+i >= r or h+i < 0 or t+j >= r or t+j < 0:
                        continue
                    grad_h += abs(I[h+i,t+j,1] - I[h,t,1])
            I[h,t,2] = grad_h



    scale_factors = 255 / np.amax(I,axis=(0,1))
    for z in range(I.shape[2]):
        for x in range(I.shape[0]):
            for y in range(I.shape[1]):
                I[x,y,z] = scale_factors[z]*I[x,y,z]
    return I#np.floor((I/r) * 256).astype(np.uint8)


def normalize(pc: pd.DataFrame):
    cols = ['x','y','z']
    pc[cols] = pc[cols].apply(lambda x: (x - x.min()), axis=0).apply(lambda x: (x/x.max()), axis=0)
    return pc

if __name__ == '__main__':    
    my_point_cloud = PyntCloud.from_file("input/mini-pc-segment-no-ground.las")
    """NORMALIZE POINTS"""
    my_point_cloud.points = normalize(my_point_cloud.points)

    sequences = TDI(my_point_cloud.points, 3)

    r = 256

    images = []

    for seq in sequences:
        images.append(Mapping_M(seq, r = r))

    print(images[0].shape)
    concatted_image = np.append(np.append(images[0], images[1], axis=1), images[2], axis=1)
    print(concatted_image.shape)
    cv2.imshow("Window", concatted_image)
    cv2.waitKey(0)