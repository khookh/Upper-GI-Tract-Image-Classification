import os
import pathlib
import random
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

study_dir = pathlib.Path(str(sys.argv[1]))

batch_size = 32
img_height = 160
img_width = 160

evals_ds = image_dataset_from_directory(
    study_dir,
    shuffle=True,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

model = tf.keras.models.load_model("./model.h5")
# model trained on kaggle with kvasir dataset

model.summary()
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
model.evaluate(evals_ds)  # evaluate on chu st pierre study set (~95% acc)

class_n = evals_ds.class_names
while True:
    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    random_dir = random.choice(os.listdir(str(sys.argv[1])))
    random_source = random.choice(os.listdir(str(sys.argv[1]) + random_dir + "/"))
    source = cv.imread(str(sys.argv[1]) + random_dir + "/" + random_source)
    img = source.copy()
    img = cv.resize(img, (160, 160))
    img = np.reshape(img, [1, 160, 160, 3])
    prediction = model.predict(img)
    print(prediction[0])
    index = np.argmax(prediction[0])
    source = cv.resize(source, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
    source = cv.putText(source, 'Pred : %s , True : %s' % (class_n[index], random_dir), (5, 20),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0),
                        1,
                        cv.LINE_AA)
    cv.imshow('proto', source)
    while (cv.waitKey(1) & 0xFF) != ord('p'):
        cv.waitKey(1)
