# importing necessary libraries
import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split,LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

#functions
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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
    df.to_csv(os.path.join(path, name + ".csv"),index=False)



def save_settings():
    setting_info = "Dataset = " + str(Dataset)
    # setting_info = setting_info + "\nvideo folder Path =" + face_dir
    setting_info = setting_info + "\naudio fodler path = " + spec_dir
    setting_info = setting_info + "\nAccuracy 1= " + str(average_acc1)
    setting_info = setting_info + "\nAccuracy 2= " + str(average_acc2)
    return setting_info


# loading the dataset
Dataset = "SAVEE"
base_dir = "/media/Data/IEEE Transaction on Affective Computing/Result/"
base1 = "VideoBase"
base2 = "AudioBase"
spec = "Audio"
face = "Video"
csv = "feature"
num_samples=6
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


# variables
class_names = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
spec_dir = os.path.join(base_dir, Dataset, base2, spec, "0.47", csv)
face_dir = os.path.join(base_dir, Dataset, base1, face, "0.92", csv)


test_spec_filename = "Test_Spec_fc6_Layer_Features.npy"
train_spec_filename = "Train_Spec_fc6_Layer_Features.npy"

test_face_filename = "Test_Face_fc6_Layer_Features.npy"
train_face_filename = "Train_Face_fc6_Layer_Features.npy"
# save_dir = os.path.join(base_dir, Dataset, base, "0_AudioVideo")

save_dir = os.path.join(base_dir, Dataset, "AudioVideo/0_AudioVideo")

test_spec_data = np.load(os.path.join(spec_dir, test_spec_filename))
train_spec_data = np.load(os.path.join(spec_dir, train_spec_filename))

test_face_data = np.load(os.path.join(face_dir, test_face_filename))
train_face_data = np.load(os.path.join(face_dir, train_face_filename))


################## Voice ###########################
################## Voice ###########################
################## Voice ###########################
################## Voice ###########################
################## Voice ###########################

data = np.vstack((train_spec_data, test_spec_data))
print(data.shape)

test_class_labels = []
for s in range(0,6):
    for l in range(0, num_samples):
        test_class_labels.append(s)

test_class_labels = np.expand_dims(test_class_labels, axis=1)
print(test_class_labels.shape)

train_class_labels = []
for s in range(0,6):
    for l in range(0, 54):
        train_class_labels.append(s)

train_class_labels = np.expand_dims(train_class_labels, axis=1)
print(train_class_labels.shape)

labels = np.vstack((train_class_labels, test_class_labels))
print(labels.shape)


##################################################
# first scenario, train on train and test on test#
##################################################
print("#####################First Scenario Started ######################")
clf = svm.SVC(kernel='linear', C=1, probability=True)
clf.fit(train_spec_data, train_class_labels)
y_predict = clf.predict(test_spec_data)
y_probs =clf.predict_proba(test_spec_data)

accuracy_score= metrics.accuracy_score(test_class_labels,y_predict) * 100
print("Accuracy 1st scenario:", accuracy_score)
conf_matrix11 = confusion_matrix(test_class_labels, y_predict)

fig, ax = plt.subplots()
im = heatmap(conf_matrix11, class_names, class_names, ax=ax, cmap="binary")
texts = annotate_heatmap(im, valfmt="{x:.0f}")
filename = os.path.join(save_dir, Dataset + "_non-normalized_confusion_matrix_audio_feature_svm_unbiased.eps")
plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


conf_matrix12 = conf_matrix11.astype('float') / conf_matrix11.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
im = heatmap(conf_matrix12, class_names, class_names, ax=ax, cmap="binary")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
filename = os.path.join(save_dir, Dataset + "_normalized_confusion_matrix_audio_feature_svm_unbiased.eps")
plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")

print("Confusion Matrix, 1st scenario")
print(conf_matrix11)
print("Confusion Matrix, 1st scenario")
print(conf_matrix12)
average_acc1 = np.average(conf_matrix12.diagonal())
save_csv(y_probs, Dataset + "_predicted_probs_audio_feature_svm_unbiased", save_dir)

print("#####################First Scenario Finished ######################")

#################################################
## second scenario, merge train and test, do 10 fold cv
#################################################
print("#####################Second Scenario Started ######################")
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)
conf_matrix21 = np.zeros((6,6))
acc = 0
probs = np.zeros((36,6))
for train, test in kfold.split(data, labels):
    clf = svm.SVC(kernel='linear', C=1, probability=True)
    training_labels = labels[train]
    clf.fit(data[train], np.ravel(training_labels, order="C"))
    clf_predictions = clf.predict(data[test])
    y_probs = clf.predict_proba(test_spec_data)
    probs = probs + y_probs
    acc += clf.score(data[test], labels[test]) * 100
    conf_matrix21 += confusion_matrix(labels[test], clf_predictions)

print("Accuracy 2nd scenario: ", (acc/10))
conf_matrix21 = conf_matrix21 / 10
probs = probs / 10

fig, ax = plt.subplots()
im = heatmap(conf_matrix21, class_names, class_names, ax=ax, cmap="binary")
texts = annotate_heatmap(im, valfmt="{x:.0f}")
filename = os.path.join(save_dir, Dataset + "_non-normalized_confusion_matrix_audio_feature_svm_biased.eps")
plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")

conf_matrix22 = conf_matrix21.astype('float') / conf_matrix21.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
im = heatmap(conf_matrix22, class_names, class_names, ax=ax, cmap="binary")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
filename = os.path.join(save_dir, Dataset + "_normalized_confusion_matrix_audio_feature_svm_biased.eps")
plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")

print("Confusion Matrix, 2nd scenario")
print(conf_matrix21)
print("Confusion Matrix, 2nd scenario")
print(conf_matrix22)
average_acc2 = np.average(conf_matrix22.diagonal())
save_csv(probs, Dataset + "_predicted_probs_audio_feature_svm_biased", save_dir)

print("#####################Second Scenario Finished ######################")

################## Voice ###########################
################## Voice ###########################
################## Voice ###########################
################## Voice ###########################
################## Voice ###########################


################## Face ###########################
################## Face ###########################
################## Face ###########################
################## Face ###########################
################## Face ###########################

data = np.vstack((train_face_data, test_face_data))
print(data.shape)

test_class_labels = []
for s in range(0,6):
    for l in range(0, num_samples):
        test_class_labels.append(s)

test_class_labels = np.expand_dims(test_class_labels, axis=1)
print(test_class_labels.shape)

train_class_labels = []
for s in range(0,6):
    for l in range(0, 54):
        train_class_labels.append(s)

train_class_labels = np.expand_dims(train_class_labels, axis=1)
print(train_class_labels.shape)

labels = np.vstack((train_class_labels, test_class_labels))
print(labels.shape)

##################################################
# first scenario, train on train and test on test#
##################################################
print("#####################First Scenario Started ######################")
clf = svm.SVC(kernel='linear', C=1, probability=True)
clf.fit(train_face_data, train_class_labels)
y_predict = clf.predict(test_face_data)
y_probs =clf.predict_proba(test_face_data)

accuracy_score= metrics.accuracy_score(test_class_labels,y_predict) * 100
print("Accuracy 1st scenario:", accuracy_score)
conf_matrix11 = confusion_matrix(test_class_labels, y_predict)

fig, ax = plt.subplots()
im = heatmap(conf_matrix11, class_names, class_names, ax=ax, cmap="binary")
texts = annotate_heatmap(im, valfmt="{x:.0f}")
filename = os.path.join(save_dir, Dataset + "_non-normalized_confusion_matrix_video_feature_svm_unbiased.eps")
plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")


conf_matrix12 = conf_matrix11.astype('float') / conf_matrix11.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
im = heatmap(conf_matrix12, class_names, class_names, ax=ax, cmap="binary")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
filename = os.path.join(save_dir, Dataset + "_normalized_confusion_matrix_video_feature_svm_unbiased.eps")
plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")

print("Confusion Matrix, 1st scenario")
print(conf_matrix11)
print("Confusion Matrix, 1st scenario")
print(conf_matrix12)
average_acc1 = np.average(conf_matrix12.diagonal())
save_csv(y_probs, Dataset + "_predicted_probs_video_feature_svm_unbiased", save_dir)

print("#####################First Scenario Finished ######################")

#################################################
## second scenario, merge train and test, do 10 fold cv
#################################################
print("#####################Second Scenario Started ######################")
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)
conf_matrix21 = np.zeros((6,6))
acc = 0
probs = np.zeros((36,6))
for train, test in kfold.split(data, labels):
    clf = svm.SVC(kernel='linear', C=1, probability=True)
    training_labels = labels[train]
    clf.fit(data[train], np.ravel(training_labels, order="C"))
    clf_predictions = clf.predict(data[test])
    y_probs = clf.predict_proba(test_face_data)
    probs = probs + y_probs
    acc += clf.score(data[test], labels[test]) * 100
    conf_matrix21 += confusion_matrix(labels[test], clf_predictions)

print("Accuracy 2nd scenario: ", (acc/10))
conf_matrix21 = conf_matrix21 / 10
probs = probs / 10

fig, ax = plt.subplots()
im = heatmap(conf_matrix21, class_names, class_names, ax=ax, cmap="binary")
texts = annotate_heatmap(im, valfmt="{x:.0f}")
filename = os.path.join(save_dir, Dataset + "_non-normalized_confusion_matrix_video_feature_svm_biased.eps")
plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")

conf_matrix22 = conf_matrix21.astype('float') / conf_matrix21.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
im = heatmap(conf_matrix22, class_names, class_names, ax=ax, cmap="binary")
texts = annotate_heatmap(im, valfmt="{x:.2f}")
filename = os.path.join(save_dir, Dataset + "_normalized_confusion_matrix_video_feature_svm_biased.eps")
plt.savefig(filename, format='eps', dpi=1000, bbox_inches="tight")

print("Confusion Matrix, 2nd scenario")
print(conf_matrix21)
print("Confusion Matrix, 2nd scenario")
print(conf_matrix22)
average_acc2 = np.average(conf_matrix22.diagonal())
save_csv(probs, Dataset + "_predicted_probs_video_feature_svm_biased", save_dir)

print("#####################Second Scenario Finished ######################")

################## Face ###########################
################## Face ###########################
################## Face ###########################
################## Face ###########################
################## Face ###########################
