import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_PATH = './cat.jpg'
LAYER_NAME = 'block5_conv3'
CAT_CLASS_INDEX = 281

img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)

# Load initial model
model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

# Create a graph that outputs target convolution and output
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

# Get the score for target class
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(np.array([img]))  # [1, 14, 14, 512], [1,1000]
    loss = predictions[:, CAT_CLASS_INDEX]  # float32


# Extract filters and gradients
output = conv_outputs[0]  # [14, 14, 512]
grads = tape.gradient(loss, conv_outputs)[0]  # [14, 14, 512] (equal shape to output)

# Average gradients spatially
weights = tf.reduce_mean(grads, axis=(0, 1))  # [512]

# Build a ponderated map of filters according to gradients importance
cam = np.ones(output.shape[0:2], dtype=np.float32)  # [14, 14]

for index, w in enumerate(weights):
    cam += w * output[:, :, index]

# Heatmap visualization
cam = cv2.resize(cam.numpy(), (224, 224))  # resize [14, 14] to [224, 224]
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())

cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)

plt.imshow(output_image)
plt.savefig('./cat_cam.png')