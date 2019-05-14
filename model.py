
from __future__ import division



import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def model(test,train,spe):

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(81, 80, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))


    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(train, epochs=5, steps_per_epoch=spe)

    test_loss, test_acc = model.evaluate(test)

    print(test_acc)




import os
from PIL import Image
import numpy
#from PIL import Image


import pathlib

def open_folder(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            path = os.path.join(root,f)
            path = os.path.abspath(path)
            paths.append(path)

    return paths
    
import numpy as np

def preprocess_image(image):
  image = tf.image.decode_png(image, channels=3)
  image = image / np.array(255, dtype=np.float32)
  return image

def load_and_preprocess_image(path, label):
  #print(path)
  image = tf.read_file(path)
  return (preprocess_image(image), label)

#print("YO")
tf.enable_eager_execution()

#open_folder()
#print(tf.__version__)

midi_dir = "./gen_midi"

#get all top level categories as path names
label_dirs = [os.path.join(midi_dir, d) for d in os.listdir(midi_dir)]

#number the labels
label_nums = {os.path.basename(d): i for i, d in enumerate(label_dirs)}
#print(label_nums)

#for each top-level category... 
paths = []
labels = []
for label_dir in label_dirs:
    print(label_dir)
    
    #...find all files in that category...
    dir_files = open_folder(label_dir)
    print(dir_files)
    paths += dir_files

    #...and label them acordingly
    label_num = label_nums[os.path.basename(label_dir)]
    print(label_num)
    for f in dir_files:
        labels.append(label_num)

#print(labels)
#print(paths)
#print(label_nums[labels[0]])

#print(labels)
#print(paths)
#print(labels)

#print(paths)

path_ds = tf.data.Dataset.from_tensor_slices(paths)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))


dataset = tf.data.Dataset.from_tensor_slices((paths, labels))


print(dataset)

DATASET_SIZE = len(paths)


ds = dataset.shuffle(buffer_size=int(DATASET_SIZE))


ds = ds.map(load_and_preprocess_image)
print(ds)

AUTOTUNE = tf.data.experimental.AUTOTUNE


BATCH_SIZE = 32
DATASET_SIZE = len(paths)

# completely shuffled.
#ds = image_label_ds.shuffle(buffer_size=int(DATASET_SIZE))
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)



steps_per_epoch= int(tf.ceil(len(paths)/BATCH_SIZE).numpy())
#print(steps_per_epoch)

#DATASET_SIZE = len(paths)
train_size = int(0.7 * DATASET_SIZE)
test_size = int(0.3 * DATASET_SIZE)

train_dataset = ds.take(train_size)
test_dataset = ds.skip(train_size)


#model(test_dataset,train_dataset,steps_per_epoch)



#print(ds)

#model(image_label_ds)

#plt.figure(figsize=(8,8))


#print(paths)
#img, labels = open_folder()

#from sklearn.model_selection import train_test_split

#train_images, test_images, train_labels, test_labels = train_test_split(img, labels, test_size=0.33, random_state=42)

#model(train_images, train_labels, test_images, test_labels)

#print("Done!")
