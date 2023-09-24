from __future__ import print_function
import time
import os
import cv2
import time
from glob import glob
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
import xlsxwriter

# mnist = fetch_mldata("MNIST original")
# X = mnist.data / 255.0
# y = mnist.target
# print(X.shape, y.shape)

class_list = ['Angry']
class_list2 = ['A']

frames = []
rng = 0
for i in range(0,1):
    base_dir = "/media/Data/Datasets/RML/Extracted_Frames/" + class_list[i] + "/Geometric Features/Landmarks_XLSX_ Files/"  # "/media/Data/Entropy/Data/Features/RML/20/9/Combined/"
    base_dir2 = "/media/Data/Datasets/RML/Extracted_Frames/" + class_list[i] + "/Geometric Features/Landmarks_XLSX_ Files/"  # "/media/Data/Entropy/Data/Kmeans/RML/20/9/"
    PATH = os.path.abspath(os.path.join(base_dir, class_list[i]))
    indices = pd.read_csv(os.path.join(base_dir2, class_list[i] + '_indices_kmean.csv'), header=None)
    categorical_label = i

    for p in range(0,10):
        # features = pd.read_csv(os.path.join(base_dir, class_list2[i] + '_' + str(p) + '.xlsx'), header=True)
        features = pd.read_excel(os.path.join(base_dir, class_list2[i] + '_' + str(p) + '.xlsx'), header=0, index_col=None)
        features = features.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        features = min_max_scaler.fit_transform(features)
        rng1 = features.shape[0]
        rng = rng + rng1
        for j in range(0,rng1):
            k = j+1
            if k in indices.iloc[p][:]:
                label = i # selected spectrograms
            else:
                label = i+6 # non-selected spectrograms
            frames.append([features[j][:], label])

print(len(frames))
X = np.array([i[0] for i in frames])
y = np.array([i[1] for i in frames])
print(X.shape, y.shape)

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))


# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

# plt.gray()
# fig = plt.figure( figsize=(16,7) )
# for i in range(0,15):
#     ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
#     ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
# plt.show()

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(6,6))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("seismic", 2),
    data=df.loc[rndperm,:],
    legend="full",
    # alpha=0.3
)

ax = plt.figure(figsize=(6,6)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"],
    ys=df.loc[rndperm,:]["pca-two"],
    zs=df.loc[rndperm,:]["pca-three"],
    c=df.loc[rndperm,:]["y"],
    cmap='prism'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()