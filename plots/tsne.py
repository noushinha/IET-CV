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
from sklearn import preprocessing
import matplotlib.patches as mpatches

base_dir  = "/media/Data/IET CV/Code/CSV_Best_Results/RML/"
base_dir2 = "/media/Data/IET CV/Results/RML/3DCNN/C3D/feature"
class_list = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
class_list2 = ['A', 'D', 'F', 'H', 'SA', 'SU']
categorical_label = 0
frames = []


# # train_face_filename = "Train_fc4_layer_features.npy"
# train_face_filename = "Train_Face_fc6_Layer_Features.npy"
# train_face_data = np.load(os.path.join(base_dir2, train_face_filename))
# pd.DataFrame(train_face_data).to_csv(os.path.join(base_dir2, "RML_Train_Face_AudioBase.csv"), index=False)

# train_face_filename = "Test_Face_fc6_Layer_Features.npy"
# train_face_data = np.load(os.path.join(base_dir2, train_face_filename))
# pd.DataFrame(train_face_data).to_csv(os.path.join(base_dir2, "RML_AudioBase_Face_Test4.csv"), index=False)

# mnist = fetch_mldata("MNIST original")
# X = mnist.data / 255.0
# y = mnist.target

labels = []
X = pd.read_csv(os.path.join(base_dir, "RML_AudioBase_Face_Train4.csv"), header=None).as_matrix().astype(np.float)
Classes = pd.read_csv(os.path.join(base_dir, "RML_VRV.csv"), header=None)
for j in range(0,Classes.shape[0]):
    if Classes[0][j] == 0:
        labels.append("Angry")
    elif Classes[0][j] == 1:
        labels.append("Disgust")
    elif Classes[0][j] == 2:
        labels.append("Fear")
    elif Classes[0][j] == 3:
        labels.append("Happiness")
    elif Classes[0][j] == 4:
        labels.append("Sadness")
    elif Classes[0][j] == 5:
        labels.append("Surprise")
# Classes = np.squeeze(Y)
print(X.shape, Classes.shape)
Classes = labels
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

N = X.shape[0]
feat_cols = ['pixel' + str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['Classes'] = Classes
df['label'] = df['Classes'].apply(lambda i: str(i))
X, Classes = None, None
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

df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values

pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
flatui = ["#9b59b6", "#3498db", "#ffaf30", "#e74c3c", "#34495e", "#2ecc71"]
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="y",
#     # palette=sns.color_palette("hls", 10),
#     palette=sns.color_palette(flatui),
#     data=df.loc[rndperm,:],
#     legend="full",
#     # alpha=0.3
# )


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

Classes = labels
df_subset['t-SNE1'] = tsne_results[:,0]
df_subset['t-SNE2'] = tsne_results[:,1]
# plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
sns.scatterplot(
    x="t-SNE1", y="t-SNE2",
    data=df_subset,
    hue="Classes",
    linewidth=.1,
    # palette=sns.color_palette("hls", 10),
    palette=sns.color_palette(flatui),
    # legend="full",
    s=28
)
ax.legend(loc='lower right', ncol=1, fontsize=9, markerscale=.7, frameon=True) # , bbox_to_anchor=(0.5, 1))
ax.set_xlabel('t-SNE1', fontdict={'fontsize': 12})
ax.set_ylabel('t-SNE2', fontdict={'fontsize': 12})
plt.title('$VR_{vnet}$', fontweight='bold', fontdict={'fontsize': 16})
plt.savefig(os.path.join(base_dir, "t-SNE_Video_VideoReferenced.eps"), format='eps', dpi=1000, bbox_inches="tight")
plt.savefig(os.path.join(base_dir, "t-SNE_Video_VideoReferenced.png"), format='png', dpi=500, bbox_inches="tight")

# plt.gca().legend().set_title('')
# recs = []
# for i in range(0,len(flatui)):
#     recs.append(mpatches.Rectangle((0,0),1,1,fc=flatui[i]))
# plt.legend(recs,class_list,loc=1)

# legend_elements = [Line2D([0], [0], marker='o', color='#9b59b6', label='Angry', markerfacecolor='#9b59b6', markersize=2),
#                    Line2D([0], [0], marker='o', color='#3498db', label='Disgust', markerfacecolor='#3498db', markersize=2),
#                    Line2D([0], [0], marker='o', color='#ffaf30', label='Fear', markerfacecolor='#ffaf30', markersize=2),
#                    Line2D([0], [0], marker='o', color='#e74c3c', label='Happiness', markerfacecolor='#e74c3c', markersize=2),
#                    Line2D([0], [0], marker='o', color='#34495e', label='Sadness', markerfacecolor='#34495e', markersize=2),
#                    Line2D([0], [0], marker='o', color='#2ecc71', label='Surprise', markerfacecolor='#2ecc71', markersize=2),
#                   ]
# ax.legend(handles=legend_elements, loc='top')
plt.show()