
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Image and bbox data
# /kaggle/input/rsna-yolo-models/1.2.826.0.1.3680043.8.498.10023411164590664678534044036963716636_1.2.826.0.1.3680043.8.498.24186535344744886473554579401056227253_slice.png
image_path = "/home/sersasj/RSNA-IAD-Codebase/data/yolo_dataset_original_fold0/images/train/1.2.826.0.1.3680043.8.498.10023411164590664678534044036963716636_1.2.826.0.1.3680043.8.498.24186535344744886473554579401056227253_slice.png"
bbox_str = "10 0.369609 0.408578 0.046875 0.046875"

# Load image
image = plt.imread(image_path)
h, w = image.shape[:2]

# Parse bbox (class_id, x_center, y_center, width, height) - all normalized
class_id, x_center, y_center, width, height = [float(x) for x in bbox_str.split()]

# Convert to pixel coordinates
x_center *= w
y_center *= h
width *= w
height *= h

# Calculate bbox corners
x1 = x_center - width/2
y1 = y_center - height/2

# Display
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
plt.gca().add_patch(Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none'))
plt.title(f'Class: {int(class_id)}')
plt.axis('off')
plt.show()