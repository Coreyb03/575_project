import kagglehub
import os
import tensorflow as tf
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from tensorflow import keras
# from keras._tf_keras.keras import layers
# from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#shap allows this image classification problem to be more 
import shap

import importFunctions as imp

def plotPredictions(labels,pred_labels,n_plots = None):
  """
  Plot pairs of images from two tensors. Each image is shown in red for true labels,
  blue for pred labels, and purple for their overlap.

  Args:
      labels (tf.Tensor): The first tensor of shape [num_images, height, width, image_features].
      pred_labels (tf.Tensor): The second tensor of shape [num_images, height, width, image_features].
      n_plots (int): Number of image pairs to plot. Defaults to all images in the tensor.
  """

  # Variables for controlling the color map for the fire masks
  CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
  BOUNDS = [-1, -0.1, 0.001, 1]
  NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)


  assert labels.shape == pred_labels.shape, "Both tensors must have the same shape"

  if n_plots is None:
    n_plots = labels.shape[0]

  fig, axes = plt.subplots(n_plots, 2, figsize=(12, 4 * n_plots))

  for i in range(n_plots):
    #actual fire mask
    axes[i,0].imshow(labels[15+i,:,:,0],cmap=CMAP,norm=NORM)
    axes[i, 0].set_title(f"Actual Fire Mask {i + 1} ")
    axes[i, 0].axis('off')

    axes[i,1].imshow(pred_labels[15+i,:,:,0],cmap=CMAP,norm=NORM)
    axes[i, 1].set_title(f"predicted Fire Mask {i + 1}")
    axes[i, 1].axis('off')

    # combined = tf.concat(labels,pred_labels)

  plt.tight_layout()
  plt.show()



print("start")
#you may need to down grade your numpy for some reason
#do pip uninstall numpy then pip install numpy==1.19.3

#this will take a while if its not installed already
path = kagglehub.dataset_download("fantineh/next-day-wildfire-spread")
print("Path to dataset files:", path)

#This should all go into notebook file in future

train_files = path + "/next_day_wildfire_spread_train*"
test_files = path + "/next_day_wildfire_spread_test*"
eval_files = path + "/next_day_wildfire_spread_eval*"



train = imp.get_dataset(train_files,  
      data_size=64,
      sample_size=32,
      batch_size=100,
      num_in_channels=12,
      compression_type=None,
      clip_and_normalize=False,
      clip_and_rescale=False,
      random_crop=True,
      center_crop=False)

test = imp.get_dataset(test_files,  
      data_size=64,
      sample_size=32,
      batch_size=100,
      num_in_channels=12,
      compression_type=None,
      clip_and_normalize=False,
      clip_and_rescale=False,
      random_crop=True,
      center_crop=False)


eval_data = imp.get_dataset(test_files,  
      data_size=64,
      sample_size=32,
      batch_size=100,
      num_in_channels=12,
      compression_type=None,
      clip_and_normalize=False,
      clip_and_rescale=False,
      random_crop=True,
      center_crop=False)


inputs, labels = next(iter(train))

titles = [
  'Elevation',
  'Wind\ndirection',
  'Wind\nvelocity',
  'Min\ntemp',
  'Max\ntemp',
  'Humidity',
  'Precip',
  'Drought',
  'Vegetation',
  'Population\ndensity',
  'Energy\nrelease\ncomponent',
  'Previous\nfire\nmask',
  'Fire\nmask'
]

output_labels = [ 'no data', 'no fire','fire']

# Number of rows of data samples to plot
n_rows = 5 
# Number of data variables
n_features = inputs.shape[3]

# Variables for controlling the color map for the fire masks
CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
BOUNDS = [-1, -0.1, 0.001, 1]
NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

#handle combined stuff soon
# combined_CMAP = colors.ListedColormap(['black', 'silver', 'red','blue','purple'])

fig = plt.figure(figsize=(15,6.5))

for i in range(n_rows):
  for j in range(n_features + 1):
    plt.subplot(n_rows, n_features + 1, i * (n_features + 1) + j + 1)
    if i == 0:
      plt.title(titles[j], fontsize=13)
    if j < n_features - 1:
      plt.imshow(inputs[i, :, :, j], cmap='viridis')
    if j == n_features - 1:
      plt.imshow(inputs[i, :, :, -1], cmap=CMAP, norm=NORM)
    if j == n_features:
      plt.imshow(labels[i, :, :, 0], cmap=CMAP, norm=NORM) 
    plt.axis('off')
plt.tight_layout()

plt.show()

#inital model, no fine tuning
#this model treats all pixels as independant, so no deep learning to look at neigbors with convolutions or anything

n_fires, height,width, feat_images = inputs.shape


#flatten the input 
x_train = tf.reshape(inputs ,[-1, feat_images]) #flatten the inputs into a 2d array, seperate each pixel as a row
y_train = tf.reshape(labels ,[-1,])


t_inputs, t_labels = next(iter(test))
# e_inputs, e_labels = next(iter(eval_data))

x_test = tf.reshape(t_inputs ,[-1, feat_images]) 
y_test = tf.reshape(t_labels ,[-1,])
# x_train = tf.reshape(inputs ,[-1, feat_images]) 
# y_train = tf.reshape(labels ,[-1,])



seed = 12
numTrees = 100 #starting number
rf = RandomForestClassifier( n_estimators= numTrees,random_state= seed,n_jobs=-1)
rf.feature_names = titles[:12]


rf.fit(x_train,y_train)
importances = rf.feature_importances_
print({rf.feature_names[i]:importances[i] for i in range(len(importances))})

#see train accuracy
y_train_pred = rf.predict(x_train)
train_acc = accuracy_score(y_train, y_train_pred)
print("Final Train Accuracy:", train_acc) #100% likely a bit overfit

y_train_pred_tensor = tf.reshape(tf.convert_to_tensor(y_train_pred), t_labels.shape)

# y_eval_pred = rf.predict(x_eval)
# eval_acc = accuracy_score(y_eval, y_eval_pred)
# print("Eval Accuracy:", eval_acc)


# Final test set evaluation
y_test_pred = rf.predict(x_test)
test_acc = accuracy_score(y_test, y_test_pred)
print("Final Test Accuracy:", test_acc)

y_pred_tensor = tf.reshape(tf.convert_to_tensor(y_test_pred), t_labels.shape)


plotPredictions(labels,y_train_pred_tensor,5)
plotPredictions(t_labels,y_pred_tensor,5)

#PLOT THE PREDICTED AND ACTUAL TEST VALUES

# Use SHAP for feature importance, this makes it interpreatable
#needs np array
x_test_np = x_test.numpy()
x_train_np = x_train.numpy()

#then to pd df to have labels
x_test_df = pd.DataFrame(x_test_np, columns = titles[:12])
x_train_df = pd.DataFrame(x_train_np, columns = titles[:12])
# print(y_test_pred.shape)

print("Tensor shape:", x_test.shape)
print("NumPy shape:", x_test_np.shape)

#limits the pixels it uses as this is slow, using the whole train/test set would take like 10 minutes
subsetSize = 5000
testSubsetSize = 10000
explainer = shap.TreeExplainer(rf, feature_names= rf.feature_names)  # Use a subset for efficiency

sampled_data_for_each_feature = {feature: x_test_df[feature].sample(n=100, random_state=seed) for feature in x_test_df.columns}
sampled_data_df = pd.concat(sampled_data_for_each_feature, axis=1)


shap_values = explainer.shap_values(sampled_data_df,check_additivity=False)


# print(shap_values.shape)
#0 is no data, 1 is no fire, 2 is fire!


shap.plots.violin(shap_values[:,:,2], features =  sampled_data_df, feature_names = titles[:12], plot_type="layered_violin")
shap.summary_plot(shap_values[:,:,2], features =  sampled_data_df, feature_names = titles[:12])



