from PIL import Image
import numpy as np
import os

# get all files
allfiles = ['male', 'female']
img_pixels = []
img_label = []
counter = 0
for a in allfiles:                     
    for ae in os.listdir(os.curdir + "/" + a):
        img_file = Image.open(os.curdir + "/" + a + "/" + ae).resize((50, 50))
        img_data = np.array(img_file.getdata())
        img_pixels.append(img_data.ravel())
        img_label.append(counter)
    counter +=1

    
 # do agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=2)
agg.fit(img_pixels)

# score
print np.sum(agg.labels_ == np.array(img_label))/float(len(img_label))
