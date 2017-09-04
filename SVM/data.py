import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

labeled_images = pd.read_csv('../input/train.csv')
set_counts = 5000

images = labeled_images.iloc[0:set_counts, 1:]
labels = labeled_images.iloc[0:set_counts, :1]

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

print(labels)