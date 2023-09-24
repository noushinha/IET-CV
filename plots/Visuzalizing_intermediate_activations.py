import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing import image
from keras import models
from keras.models import model_from_json
import os

base_dir   = "/media/Data/Entropy/Data/npy/RML/20/9/Train_Test_Validation/"

#load data
train_data = np.load(os.path.join(base_dir, "spec_train_data_ordered.npy"))

# load json and create model
json_file = open('/media/Data/Entropy/Result/RML/20/9/0.72/RML_Spec_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('/media/Data/Entropy/Result/RML/20/9/0.72/weights-improvement-014-0.57-1.15.hdf5')
# print(model.summary())

img_tensor = train_data[0]  # '/media/Data/Entropy/Data/KeySpectrograms/RML/20/9/train/Angry/A_000_001.png'
img_tensor = np.expand_dims(img_tensor, axis=0)
# img = image.load_img(img_path, target_size=(96, 96))
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255.

print(img_tensor.shape)
img=train_data[0]
plt.imshow(img[0])
#plt.figure()

#extracts the outputs of the top eight layer
layer_outputs = [layer.output for layer in model.layers[:8]]
#creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
#returns a list of five numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, :, 3], cmap='viridis')
plt.show()

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]


    size = layer_activation.shape[1]
    size2 = layer_activation.shape[2]
    size3 = layer_activation.shape[4]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # for immg in range(10):
    #     if immg != 0:
    #         break
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            if channel_image.std() != 0:
                channel_image /= channel_image.std()
            else:
                channel_image /= 1.
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size2 : (col + 1) * size2, row * size3 : (row + 1) * size3] = channel_image[0]

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()