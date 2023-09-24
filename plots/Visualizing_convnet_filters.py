import os
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import  VGG16
from keras import backend as K
from keras.models import model_from_json

#Defining loss tensor for filter visualization
# model = VGG16(weights='imagenet',
#               include_top=False)
# layer_name = 'block3_conv1'


json_file = open('/media/Data/Entropy/Result/RML/20/9/0.72/RML_Spec_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('/media/Data/Entropy/Result/RML/20/9/0.72/weights-improvement-014-0.57-1.15.hdf5')
layer_name = 'conv3d_2'
print(model.summary())

base_dir   = "/media/Data/Entropy/Data/npy/RML/20/9/Train_Test_Validation/"

#load data
train_data = np.load(os.path.join(base_dir, "spec_train_data_ordered.npy"))
train_data = train_data.astype('float32')
filter_index = 0
size = 96
margin = 5

# Utility function to convert a tensor into a valid image
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=48):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Obtaining the gradient of the oss with regard to the input
    grads = K.gradients(loss, model.input)[0]

    # Gradient-Normalization trick
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # Fetching Numpy output values given Numpy input values
    iterate = K.function([model.input], [loss, grads])

    #loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

    # Loss Maximization via stochastic gradient descent
    img_data = np.expand_dims(train_data[0], axis=0)
    input_img_data = img_data  # np.random.random((1, 9, 96, 96, 3)) * 20 + 128

    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])

        input_img_data += grads_value * step
        # input_img_data = np.add(input_img_data, (grads_value * step), out=input_img_data, casting="unsafe")

    img = input_img_data[0]
    return deprocess_image(img)


results = np.zeros((5 * size + 7 * margin, 5 * size + 7 * margin, 3))

for i in range(5):
    for j in range(5):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start:horizontal_end,
                vertical_start:vertical_end, :] = filter_img[j]

plt.figure(figsize=(10, 10))
plt.imshow(results)
plt.show()
#plt.imshow(generate_pattern(layer_name, filter_index))
#plt.show()



