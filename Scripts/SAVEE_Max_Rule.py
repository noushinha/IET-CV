import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
import matplotlib
import matplotlib.pyplot as plt


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=True, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def save_csv(data, name, path):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path, name + ".csv"))


def save_settings():
    setting_info = "Dataset = " + str(Dataset)
    setting_info = setting_info + "\nvideo folder Path =" + face_dir
    setting_info = setting_info + "\naudio fodler path = " + spec_dir
    setting_info = setting_info + "\nAccuracy = " + str(average_acc)
    return setting_info


Dataset = "SAVEE"
base_dir = "/media/Data/IEEE Transaction on Affective Computing/Result/"
base1 = "VideoBase"
base2 = "AudioBase"
spec = "Audio"
face = "Video"
csv = "csv"
num_samples=6
class_names = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
spec_dir = os.path.join(base_dir, Dataset, base2, spec, "0.47", csv)
face_dir = os.path.join(base_dir, Dataset, base1, face, "0.92", csv)

test_spec_filename = "Test_Spec_Probabilities_155_Epochs.csv"
test_face_filename = "Test_Face_Probabilities_55_Epochs.csv"
# save_dir = os.path.join(base_dir, Dataset, base, "0_AudioVideo")
save_dir = os.path.join(base_dir, Dataset, "AudioVideo/0_AudioVideo")

spec_data = pd.read_csv(os.path.join(spec_dir, test_spec_filename))
face_data = pd.read_csv(os.path.join(face_dir, test_face_filename))

print(spec_data.shape, face_data.shape)

spec_data = spec_data.as_matrix()
face_data = face_data.as_matrix()

spec_data = spec_data * .38
face_data = face_data * .62

data = np.add(spec_data, face_data)

labels = np.argmax(data, axis=1)
labels = labels.reshape(6,num_samples)
print(labels)

conf_matrix = np.zeros((6,6))
for i in range(0,6):
    for j in range(0,num_samples):
        if(labels[i][j] == i):
            conf_matrix[i][i] = conf_matrix[i][i] + 1
        else:
            k = labels[i][j]
            conf_matrix[i][k] = conf_matrix[i][k] + 1


fig, ax = plt.subplots()
im = heatmap(conf_matrix, class_names, class_names, ax=ax, cmap="binary")
texts = annotate_heatmap(im, valfmt="{x:.0f}")
filename = os.path.join(save_dir, Dataset + "_non-normalized_confusion_matrix_max_rule.eps")
plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


conf_matrix2 = (conf_matrix /num_samples) * 100
fig, ax = plt.subplots()
im = heatmap(conf_matrix2, class_names, class_names, ax=ax, cmap="binary")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
filename = os.path.join(save_dir, Dataset + "_normalized_confusion_matrix_max_rule.eps")
plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")
# plt.show( block=True)

print(conf_matrix)
print(conf_matrix2)
average_acc = np.average(conf_matrix2.diagonal())

save_csv(conf_matrix, Dataset + "_non-normalized_confusion_matrix_max_rule", save_dir)
save_csv(conf_matrix2, Dataset + "_normalized_confusion_matrix_max_rule", save_dir)
save_csv(labels, Dataset + "_predicted_labels_max_rule", save_dir)
HyperParameter_Setting = save_settings()
with open(os.path.join(save_dir, Dataset + "_results_max_rule.txt"), "w") as text_file:
    text_file.write(HyperParameter_Setting)

print(HyperParameter_Setting)


