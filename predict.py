from main import *
from util import *

resolution = 128
merged_trees = PyntCloud.from_file("test_trees/subareas/" + "sub1" + ".las")
pred_trees = PyntCloud.from_file("test_trees/subareas/" + "sub1" + ".las")


merged_trees.points = normalize(merged_trees.points)
print(pred_trees.points)
print(merged_trees.points)
sequences = TDI(merged_trees.points, 3)
image = Mapping_M(sequences[-1],  r = resolution)

#RUN THE IMAGE THROUGH THE PREDICTOR; GET BACK A BUNCH OF BOXES WITH THRESHES:
preds = prediction(image)

# Create a grid with points that falls within each pixels
for pred in preds:
    # check pred thresh > 70%
    # Get list of pixels inside predbox/segmentation poly
    for pixel in pred:
        for point in pixel:
            ""
            # point["tree_instance"] = id
    
# return image