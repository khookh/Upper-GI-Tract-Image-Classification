import tensorflow as tf
import sys
import pathlib
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print(tf.__version__)

data_dir = pathlib.Path(str(sys.argv[1]))
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 180
img_width = 180

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

print(train_ds.class_names)

# TEST
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis("off")
plt.show()
# TEST
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)