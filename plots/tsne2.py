from __future__ import print_function
import os
import cv2
import time
from glob import glob
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

base_dir = "/media/Data/Datasets/SAVEE/Extracted_frames/Emotion Categories/"
base_dir2 = "/media/Data/Entropy/Data/Kmeans/SAVEE/20/9/"
class_list = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
class_list2 = ['A', 'D', 'F', 'H', 'SA', 'SU']
WIDTH = 28
HEIGHT = 28
frames = []
for i in range(0, 1):
    PATH = os.path.abspath(os.path.join(base_dir, class_list[i]))
    indices = pd.read_csv(os.path.join(base_dir2, class_list[i], class_list[i] + '_indices_kmean.csv'), header=None)

    images = glob(os.path.join(PATH, "*_0004_*.png"))
    images.sort(key=lambda f: int(filter(str.isdigit, f)))
    for img in images:
        full_size_image = cv2.imread(img) / 255
        color_image = cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY).reshape(784)

        base = os.path.basename(img)
        indx = base.split("_")
        p = int(indx[1])
        k = int(indx[2].split(".")[0])
        if k in indices.iloc[p][:]:
            label = "Selected"  # i selected spectrograms
        else:
            label = "Non-selected"  # i+1 # non-selected spectrograms
        frames.append([gray_image, label])


print(len(frames))
X = np.array([i[0] for i in frames])
y = np.array([i[1] for i in frames])
print(X.shape, y.shape)
feat_cols = ['pixel' + str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X, columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))


# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

N = df.shape[0]
df_subset = df.loc[rndperm[:N], :].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:, 0]
df_subset['pca-two'] = pca_result[:, 1]
df_subset['pca-three'] = pca_result[:, 2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:, 0]
df_subset['tsne-2d-two'] = tsne_results[:, 1]
plt.figure(figsize=(5, 5))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("colorblind", 2),
    data=df_subset,
    legend="full"
)
plt.show()