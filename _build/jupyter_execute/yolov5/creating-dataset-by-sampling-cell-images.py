#!/usr/bin/env python
# coding: utf-8

# # Counting cells using YOLOv5
# 
# YOLOv5: https://github.com/ultralytics/yolov5
# 
# ---
# Author of this notebook: Andre Telfer (andretelfer@cmail.carleton.ca)

# In[1]:


pip install shapely


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
import napari
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from skimage.io import imread, imsave


# ## What does our dataset look like?
# Our dataset is just a folder containing `.png` images of cfos stains

# In[3]:


DATA_DIR = Path("/home/andretelfer/shared/curated/brenna/cfos-examples/original")


# In[4]:


plt.figure(figsize=(20,20))
sample_image = next(DATA_DIR.glob('*.png'))
image = imread(sample_image)
plt.imshow(image)


# ## Create a new dataset by sampling sections of the original dataset

# In[5]:


SIZE = 200 # size of new images in pixels
SAMPLES_PER_IMAGE = 5

def subsample_image(imagepath, samples):
    image = imread(imagepath)
    h,w,c = image.shape
    locations = np.stack([
        np.random.randint(0,h-SIZE,samples),
        np.random.randint(0,w-SIZE,samples)
    ]).T
    
    images = []
    for (i,j) in locations:
        new_image = np.zeros(shape=(SIZE,SIZE))
        new_image = image[i:i+SIZE, j:j+SIZE]
        images.append({
            'image': new_image,
            'x': j,
            'y': i,
            'path': imagepath
        })
        
    return images

sampled_images = []
for image in DATA_DIR.glob('*.png'):
    sampled_images += subsample_image(image, SAMPLES_PER_IMAGE)
    
plt.imshow(sampled_images[5]['image'])


# ## Save the images to a new directory 

# In[6]:


OUTPUT_DIR = Path("/home/andretelfer/shared/curated/brenna/cfos-examples/subsamples")

get_ipython().system(' rm -rf {OUTPUT_DIR}')
get_ipython().system(' mkdir -p {OUTPUT_DIR}')

for item in sampled_images:
    image = item['image']
    fname = item['path'].parts[-1].split('.')[0]
    
    output_file = f"{fname}_{item['x']}x_{item['y']}y.png"
    imsave(OUTPUT_DIR / output_file, image)


# In[7]:


ls {OUTPUT_DIR}


# ## Labeling the Images
# For this step, the [YOLOv5 documentation](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) recommended [roboflow](https://app.roboflow.com/).
# 
# I created a free-tier account and started labelling. 
# 
# I modified the tutorial by not including a scaling preprocessing step. I did this because the size of the cell matters and I wanted to preserve that information
# - there are some large splotches that are cell shaped, but are not cells
# - there are small speckles which are not cells
# 
# 
# ![](./assets/screenshot-roboflow.png)

# ## Training a Model
# The YOLOv5 guide came with a [Google Colab notebook](https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb) that was easy to modify to my own examples (dataset, image sizes, etc)
# - Following the guide, I changed the dataset to the one we created in RoboFlow
# - In order to preserve information about cell size, for training I set the image size to be the same as the actual image size for the sampled training images. When running inference/detection, I used the full image size (e.g. ~2000px in my case).
# 
# ![](./assets/screenshot-colab.png)

# ## Inference and Results
# I uploaded the original images to google drive separately and modified the notebook to use them
# 
# The results were overall quite good, although I later found I should've labeled more of the lighter cells in the training data; so the model also misses the lighter cells.
# 
# ![](./assets/yolov5-results/Rat2slide1sample3-L.png)

# ## Interacting with the results (correcting/viewing)
# This will allow us to add/remove cells that were missed
# ![](./assets/screenshot-napari-cells.png)

# ### Loading the YOLO labels

# In[8]:


all_vertices = []

for idx, p in enumerate(sorted(LABEL_DIR.glob("*.txt"))):
    with open(p, 'r') as fp:
        lines = fp.readlines()
    
    # Turn the data into a dataframe
    data = [l.strip().split(' ') for l in lines]
    df = pd.DataFrame(data, columns=['0', 'x', 'y', 'w', 'h', 'c']).astype(float)

    # Drop low confidence frames
    df = df[df.c > 0.2]
    
    # Scale by image size
    fname, _ = p.parts[-1].split('.')
    image = imread(DATA_DIR / f"{fname}.png")
    h, w, c = image.shape
    df.y *= h
    df.h *= h
    df.x *= w
    df.w *= w
    
    # Get x,y vertices for rectangle
    i = np.ones(shape=df.shape[0])*idx
    vertices = np.array([
        [i, df.y-df.h/2, df.x-df.w/2],
        [i, df.y+df.h/2, df.x-df.w/2],
        [i, df.y+df.h/2, df.x+df.w/2],
        [i, df.y-df.h/2, df.x+df.w/2]
    ]).transpose(2, 0, 1)
    all_vertices.append(vertices)
    
all_vertices = np.concatenate(all_vertices)


# ### Viewing them with Napari

# In[ ]:



# The YOLOv5 labels
LABEL_DIR = Path("assets/yolov5-results/labels")

viewer = napari.Viewer()

# Add the images
images = np.array([imread(p) for p in sorted(DATA_DIR.glob("*.png"))])
image_layer = viewer.add_image(np.array(images))

# Add the yolov5 labels
shape_layer = viewer.add_shapes(all_vertices, face_color=[1., 0., 0., 0.3])


# ## Getting cell counts

# In[116]:


shape_layer.save('assets/cells.csv')
cells_df = pd.read_csv('assets/cells.csv')
cells_df.head(3)


# In[127]:


cells_df = cells_df.rename(columns={'axis-0': 'image', 'axis-1': 'y', 'axis-2' : 'x', 'index': 'cell'})
cells_df.head(5)


# Finally, we can get the cell counts for each image

# In[169]:


cell_counts = cells_df.groupby('image').apply(lambda x: len(x.cell.unique()))


# ## Getting cells in an area
# ![](./assets/screenshot-napari-zone.png)

# In[ ]:


zone_layer = viewer.add_shapes(name='zone', ndim=3, edge_color='red', face_color=[0.,0.,1.,0.3])


# In[174]:


zone_layer.save('assets/zones.csv')
zone_df = pd.read_csv('assets/zones.csv')
zone_df = zone_df.rename(columns={'axis-0': 'image', 'axis-1': 'y', 'axis-2' : 'x'})

cells_by_image = cells_df.groupby('image')
zones_by_image = zone_df.groupby('image')

for (idx, zone), (idx, cells) in zip(zones_by_image, cells_by_image):
    zone = shapely.geometry.Polygon(zone[['x', 'y']].values)
    
    plt.figure(figsize=(16,16))
    plt.imshow(images[int(idx)])
    x,y = zone.exterior.xy
    plt.plot(x,y)
    ax = plt.gca()
    
    count = 0
    for _, cell in tqdm(cells.groupby('cell')):
        cell = shapely.geometry.Polygon(cell[['x', 'y']].values)
        if zone.contains(cell):
            count += 1
            ax.add_patch(plt.Polygon(np.stack(cell.exterior.xy).T, color='red'))
    
    plt.show()
    
    print("Cell count", count)
    print("Density", count / zone.area * 1e6)
    print()

