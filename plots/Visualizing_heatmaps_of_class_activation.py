import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import backend as K

model = VGG16(weights='imagenet')

# Local path to the target image
img_path = '/media/Data/Entropy/Data/KeySpectrograms/RML/20/9/train/Angry/A_000_004.png'

# Python Image Library (PIL) image of size 224 x 224 for VGGT16
img = image.load_img(img_path, target_size=(224, 224))

# Converting to float32 nNumpy Array of shape (224, 224, 3)
x = image.img_to_array(img)

# Adds a dimension to transform the array into a batch of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Preprocesses the batch(this does channel-wise color normalization)
x = preprocess_input(x)

preds = model.predict(x)
print('predicted: ', decode_predictions(preds, top=3)[0])

# finding the index of african elephant
print(np.argmax(preds[0]))

# African Elephant entry in the prediction vector
african_elephant_output = model.output[:, 386]

# Output feature map of the blockj5_conv3 layer, the last convolutional layer in VGG16
last_conv_layer = model.get_layer('block5_conv3')

# Gradient of the "African elephant" class with regard to the output feature map of block5_conv3
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# Vector of shape (512,), where each entry is the mean intensity of the gradient over a specific feature-map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# Lets me access the values of the quantities you just defined:
# pooled_grads and the output feature map of block5_conv3, givena  sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# Values of these two quantities, as Numpy arrays, given the sample image of two elephants
pooled_grad_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grad_value[i]

heatmap = np.mean(conv_layer_output_value, axis =-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

# Loading the original image using cv2
img = cv2.imread(img_path)

# Resizes the heatmap to be the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# Converts the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# Applies the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# heatmap intensity factor is 0.4
superimposed_img = heatmap * 0.4 + img

# Saving the image into the disk
cv2.imwrite('/home/deeplearning/PycharmProjects/fchallote/elephant_CAM.jpg', superimposed_img)