---
layout: post
title: Image Classification
---
In this blog post, you will learn several new skills and concepts related to image classification in Tensorflow.

Tensorflow Datasets provide a convenient way for us to organize operations on our training, validation, and test data sets.
Data augmentation allows us to create expanded versions of our data sets that allow models to learn patterns more robustly.
Transfer learning allows us to use pre-trained models for new tasks.
Working on the coding portion of the Blog Post in Google Colab is strongly recommended. When training your model, enabling a GPU runtime (under Runtime -> Change Runtime Type) is likely to lead to significant speed benefits.

# The Task

Can you teach a machine learning algorithm to distinguish between pictures of dogs and pictures of cats?

According to this helpful diagram below, one way to do this is to attend to the visible emotional range of the pet:

## §1. Load Packages and Obtain Data

Start by making a code block in which you’ll hold your import statements. You can update this block as you go. For now, include

```python
import os
import tensorflow as tf
from tensorflow.keras import utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

Now, let’s access the data. We’ll use a sample data set provided by the TensorFlow team that contains labeled images of cats and dogs.

Paste and run the following code block.

```python
# location of data
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# download the data and extract it
path_to_zip = utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# construct paths
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# parameters for datasets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# construct train and validation datasets 
train_dataset = utils.image_dataset_from_directory(train_dir,
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   image_size=IMG_SIZE)

validation_dataset = utils.image_dataset_from_directory(validation_dir,
                                                        shuffle=True,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMG_SIZE)

# construct the test dataset by taking every 5th observation out of the validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
```

    Found 2000 files belonging to 2 classes.
    Found 1000 files belonging to 2 classes.

By running this code, we have created TensorFlow Datasets for training, validation, and testing. You can think of a Dataset as a pipeline that feeds data to a machine learning model. We use data sets in cases in which it’s not necessarily practical to load all the data into memory.

In our case, we’ve used a special-purpose keras utility called image_dataset_from_directory to construct a Dataset. The most important argument is the first one, which says where the images are located. The shuffle argument says that, when retrieving data from this directory, the order should be randomized. The batch_size determines how many data points are gathered from the directory at once. Here, for example, each time we request some data we will get 32 images from each of the data sets. Finally, the image_size specifies the size of the input images, just like you’d expect.

## Working with Datasets

You can get a piece of a data set using the take method; e.g. train_dataset.take(1) will retrieve one batch (32 images with labels) from the training data.

Let’s briefly explore our data set. We will write a function to create a two-row visualization. In the first row, it shows three random pictures of cats. In the second row, it shows three random pictures of dogs. You can see some related code in the linked tutorial above, although you’ll need to make some modifications in order to separate cats and dogs by rows. A docstring is not required.

```python
def plot_images(dataset):
    class_names = dataset.class_names
    plt.figure(figsize= (10, 6))
    for images, labels in dataset.take(1):
        x = 0
        for i in range(len(images)):
            if(x < 3):
                if(class_names[labels[i]] == "cats"):
                    ax = plt.subplot(2, 3, x + 1)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title(class_names[labels[i]])
                    plt.axis("off")
                    x += 1
            if(x >= 3):
                if(class_names[labels[i]] == "dogs"):
                    ax = plt.subplot(2, 3, x + 1)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title(class_names[labels[i]])
                    plt.axis("off")
                    x += 1
            if(x >= 6):
                break
plot_images(train_dataset)
```


![](https://github.com/jameswest25/jameswest25.github.io/blob/master/images/HW3_Image_Classification_files/HW3_Image_Classification_2_0.png?raw=true)
    

Paste the following code into the next block. This is technical code related to rapidly reading data. 

```python
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
```
### Check Label Frequencies
The first line of code will create an iterator called labels.
This code compute the number of images in the training data with label 0 (corresponding to "cat") and label 1 (corresponding to "dog").

The baseline machine learning model is the model that always guesses the most frequent label. 

We’ll treat this as the benchmark for improvement. Our models should do much better than baseline in order to be considered good data science achievements!

```python
labels_iterator= train_dataset.unbatch().map(tf.autograph.experimental.do_not_convert(lambda image, label: label)).as_numpy_iterator()
catcount, dogcount = 0, 0
for x in labels_iterator:
    if x == 0:
        catcount += 1
    else:
        dogcount += 1
print(catcount)
print(dogcount)
#The baseline model would be accurate 50% of the time
```

    1000
    1000

## §2. First Model
Now, we wil create a tf.keras.Sequential model. In each model, we include at least two Conv2D layers, at least two MaxPooling2D layers, at least one Flatten layer, at least one Dense layer, and at least one Dropout layer. We train the model and plot the history of the accuracy on both the training and validation sets. 

```python
num_classes = 2
model1 = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model1.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history1 = model1.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 34s 522ms/step - loss: 0.7239 - accuracy: 0.5210 - val_loss: 0.6792 - val_accuracy: 0.6027
    Epoch 2/20
    63/63 [==============================] - 33s 517ms/step - loss: 0.6640 - accuracy: 0.6115 - val_loss: 0.6716 - val_accuracy: 0.6225
    Epoch 3/20
    63/63 [==============================] - 32s 499ms/step - loss: 0.6197 - accuracy: 0.6545 - val_loss: 0.6372 - val_accuracy: 0.6671
    Epoch 4/20
    63/63 [==============================] - 35s 549ms/step - loss: 0.5611 - accuracy: 0.7145 - val_loss: 0.6400 - val_accuracy: 0.6547
    Epoch 5/20
    63/63 [==============================] - 35s 561ms/step - loss: 0.4936 - accuracy: 0.7635 - val_loss: 0.6452 - val_accuracy: 0.6510
    Epoch 6/20
    63/63 [==============================] - 32s 500ms/step - loss: 0.4242 - accuracy: 0.8035 - val_loss: 0.6914 - val_accuracy: 0.6275
    Epoch 7/20
    63/63 [==============================] - 32s 511ms/step - loss: 0.3456 - accuracy: 0.8465 - val_loss: 0.6891 - val_accuracy: 0.6535
    Epoch 8/20
    63/63 [==============================] - 32s 506ms/step - loss: 0.2544 - accuracy: 0.8935 - val_loss: 0.7725 - val_accuracy: 0.6609
    Epoch 9/20
    63/63 [==============================] - 31s 488ms/step - loss: 0.2021 - accuracy: 0.9185 - val_loss: 0.8018 - val_accuracy: 0.6968
    Epoch 10/20
    63/63 [==============================] - 32s 499ms/step - loss: 0.1331 - accuracy: 0.9475 - val_loss: 0.9344 - val_accuracy: 0.6584
    Epoch 11/20
    63/63 [==============================] - 33s 516ms/step - loss: 0.0840 - accuracy: 0.9705 - val_loss: 1.0224 - val_accuracy: 0.6547
    Epoch 12/20
    63/63 [==============================] - 31s 488ms/step - loss: 0.0895 - accuracy: 0.9705 - val_loss: 1.0745 - val_accuracy: 0.6411
    Epoch 13/20
    63/63 [==============================] - 31s 495ms/step - loss: 0.0587 - accuracy: 0.9820 - val_loss: 1.1281 - val_accuracy: 0.6287
    Epoch 14/20
    63/63 [==============================] - 30s 473ms/step - loss: 0.0650 - accuracy: 0.9735 - val_loss: 1.0347 - val_accuracy: 0.6696
    Epoch 15/20
    63/63 [==============================] - 31s 496ms/step - loss: 0.0357 - accuracy: 0.9915 - val_loss: 1.2658 - val_accuracy: 0.6671
    Epoch 16/20
    63/63 [==============================] - 35s 550ms/step - loss: 0.0276 - accuracy: 0.9910 - val_loss: 1.3278 - val_accuracy: 0.6782
    Epoch 17/20
    63/63 [==============================] - 36s 563ms/step - loss: 0.0344 - accuracy: 0.9905 - val_loss: 1.0871 - val_accuracy: 0.6881
    Epoch 18/20
    63/63 [==============================] - 33s 518ms/step - loss: 0.0336 - accuracy: 0.9900 - val_loss: 1.2981 - val_accuracy: 0.6646
    Epoch 19/20
    63/63 [==============================] - 34s 531ms/step - loss: 0.0178 - accuracy: 0.9965 - val_loss: 1.5740 - val_accuracy: 0.6522
    Epoch 20/20
    63/63 [==============================] - 51s 805ms/step - loss: 0.0274 - accuracy: 0.9910 - val_loss: 1.2849 - val_accuracy: 0.6584



My model stabilized between 68-73% accuracy. This is better than the baseline of 50% accuracy. However, considering the difference between training accuracy(~95%) and validation accuracy(~71%), it is clear
that overfitting occured with this model.

We will now plot the model. 


```python
def plot_model(history):
    model_history = pd.DataFrame(history.history)
    model_history['epoch'] = history.epoch
    fig, ax = plt.subplots(1, figsize=(8,6))
    num_epochs = model_history.shape[0]
    ax.plot(np.arange(0, num_epochs), model_history["accuracy"], 
        label="Training Accuracy")
    ax.plot(np.arange(0, num_epochs), model_history["val_accuracy"], 
        label="Validation Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.show()
plot_model(history1)
```


    
![](https://github.com/jameswest25/jameswest25.github.io/blob/master/images/HW3_Image_Classification_files/HW3_Image_Classification_7_0.png?raw=true)
    
    
## §3. Model with Data Augmentation
Now we’re going to add some data augmentation layers to your model. Data augmentation refers to the practice of including modified copies of the same image in the training set. For example, a picture of a cat is still a picture of a cat even if we flip it upside down or rotate it 90 degrees. We can include such transformed versions of the image in our training process in order to help our model learn so-called invariant features of our input images. Here are examples of randomflip and randomrotation layers.


```python
image,labels = next(iter(train_dataset))
random_flip = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=.5)
])
plt.figure(figsize=(10, 3))
ax = plt.subplot(1, 3, 1)
plt.imshow(image[0,:,:,0])
plt.axis("off")
for i in range(2):
    augmented_image = random_flip(image)
    ax = plt.subplot(1, 3, i + 2)
    plt.imshow(augmented_image[0,:,:,0])
    plt.axis("off")
```

![](https://github.com/jameswest25/jameswest25.github.io/blob/master/images/HW3_Image_Classification_files/HW3_Image_Classification_8_0.png?raw=true)
    


```python
image, label = next(iter(train_dataset))
random_rotation = tf.keras.Sequential([
   tf.keras.layers.RandomRotation(0.7, seed=1)
])
plt.figure(figsize=(10, 3))
ax = plt.subplot(1, 3, 1)
plt.imshow(image[0,:,:,0])
plt.axis("off")
for i in range(2):
    augmented_image = random_rotation(image)
    ax = plt.subplot(1, 3, i + 2)
    plt.imshow(augmented_image[0,:,:,0])
    plt.axis("off")
    image = augmented_image
```


    
![](https://github.com/jameswest25/jameswest25.github.io/blob/master/images/HW3_Image_Classification_files/HW3_Image_Classification_9_0.png?raw=true)
    

Now, we will create our model with the data augmentation layers. 

```python
model2 = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
model2.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history2 = model2.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 38s 588ms/step - loss: 0.7398 - accuracy: 0.5180 - val_loss: 0.6914 - val_accuracy: 0.5297
    Epoch 2/20
    63/63 [==============================] - 36s 566ms/step - loss: 0.6922 - accuracy: 0.5365 - val_loss: 0.6874 - val_accuracy: 0.5755
    Epoch 3/20
    63/63 [==============================] - 35s 560ms/step - loss: 0.6891 - accuracy: 0.5400 - val_loss: 0.6922 - val_accuracy: 0.5074
    Epoch 4/20
    63/63 [==============================] - 34s 545ms/step - loss: 0.6918 - accuracy: 0.5140 - val_loss: 0.6863 - val_accuracy: 0.5198
    Epoch 5/20
    63/63 [==============================] - 34s 539ms/step - loss: 0.6905 - accuracy: 0.5195 - val_loss: 0.6888 - val_accuracy: 0.5483
    Epoch 6/20
    63/63 [==============================] - 34s 538ms/step - loss: 0.6918 - accuracy: 0.5405 - val_loss: 0.6890 - val_accuracy: 0.5210
    Epoch 7/20
    63/63 [==============================] - 41s 649ms/step - loss: 0.6846 - accuracy: 0.5395 - val_loss: 0.6869 - val_accuracy: 0.5520
    Epoch 8/20
    63/63 [==============================] - 37s 583ms/step - loss: 0.6722 - accuracy: 0.5825 - val_loss: 0.6740 - val_accuracy: 0.5941
    Epoch 9/20
    63/63 [==============================] - 39s 616ms/step - loss: 0.6674 - accuracy: 0.5980 - val_loss: 0.6689 - val_accuracy: 0.6114
    Epoch 10/20
    63/63 [==============================] - 40s 631ms/step - loss: 0.6619 - accuracy: 0.6065 - val_loss: 0.6543 - val_accuracy: 0.6200
    Epoch 11/20
    63/63 [==============================] - 37s 583ms/step - loss: 0.6435 - accuracy: 0.6385 - val_loss: 0.6700 - val_accuracy: 0.5903
    Epoch 12/20
    63/63 [==============================] - 33s 525ms/step - loss: 0.6319 - accuracy: 0.6470 - val_loss: 0.6381 - val_accuracy: 0.6188
    Epoch 13/20
    63/63 [==============================] - 33s 527ms/step - loss: 0.6093 - accuracy: 0.6720 - val_loss: 0.6422 - val_accuracy: 0.6349
    Epoch 14/20
    63/63 [==============================] - 34s 530ms/step - loss: 0.6163 - accuracy: 0.6575 - val_loss: 0.6118 - val_accuracy: 0.6460
    Epoch 15/20
    63/63 [==============================] - 33s 524ms/step - loss: 0.6184 - accuracy: 0.6635 - val_loss: 0.6117 - val_accuracy: 0.6671
    Epoch 16/20
    63/63 [==============================] - 34s 534ms/step - loss: 0.6047 - accuracy: 0.6700 - val_loss: 0.6219 - val_accuracy: 0.6547
    Epoch 17/20
    63/63 [==============================] - 33s 527ms/step - loss: 0.5825 - accuracy: 0.6890 - val_loss: 0.6096 - val_accuracy: 0.6287
    Epoch 18/20
    63/63 [==============================] - 40s 632ms/step - loss: 0.5730 - accuracy: 0.7075 - val_loss: 0.5977 - val_accuracy: 0.6708
    Epoch 19/20
    63/63 [==============================] - 40s 628ms/step - loss: 0.5772 - accuracy: 0.7005 - val_loss: 0.5810 - val_accuracy: 0.7042
    Epoch 20/20
    63/63 [==============================] - 38s 604ms/step - loss: 0.5531 - accuracy: 0.7270 - val_loss: 0.5828 - val_accuracy: 0.6968



My model stabilized between 68-73% accuracy. This is pretty similar to model1. However, considering the difference between training accuracy(~72%) and validation accuracy(~71%), it is clear that there was less overfitting with this model.

Now we will plot the accuracy of the model. 


```python
plot_model(history2)
```


    
![](https://github.com/jameswest25/jameswest25.github.io/blob/master/images/HW3_Image_Classification_files/HW3_Image_Classification_12_0.png?raw=true)
    

## §4. Data Preprocessing
Sometimes, it can be helpful to make simple transformations to the input data. For example, in this case, the original data has pixels with RGB values between 0 and 255, but many models will train faster with RGB values normalized between 0 and 1, or possibly between -1 and 1. These are mathematically identical situations, since we can always just scale the weights. But if we handle the scaling prior to the training process, we can spend more of our training energy handling actual signal in the data and less energy having the weights adjust to the data scale.

The following code will create a preprocessing layer called preprocessor which we can slot into our model pipeline.

```python
i = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
preprocessor = tf.keras.Model(inputs = [i], outputs = [x])

model3 = tf.keras.Sequential([
    preprocessor,
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
model3.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history3 = model3.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 34s 516ms/step - loss: 0.7111 - accuracy: 0.5330 - val_loss: 0.6820 - val_accuracy: 0.5606
    Epoch 2/20
    63/63 [==============================] - 33s 526ms/step - loss: 0.6574 - accuracy: 0.6040 - val_loss: 0.6599 - val_accuracy: 0.5347
    Epoch 3/20
    63/63 [==============================] - 33s 524ms/step - loss: 0.6324 - accuracy: 0.6395 - val_loss: 0.6204 - val_accuracy: 0.6225
    Epoch 4/20
    63/63 [==============================] - 33s 522ms/step - loss: 0.6240 - accuracy: 0.6470 - val_loss: 0.6166 - val_accuracy: 0.6757
    Epoch 5/20
    63/63 [==============================] - 33s 517ms/step - loss: 0.5966 - accuracy: 0.6700 - val_loss: 0.6020 - val_accuracy: 0.6856
    Epoch 6/20
    63/63 [==============================] - 33s 518ms/step - loss: 0.5857 - accuracy: 0.6795 - val_loss: 0.6056 - val_accuracy: 0.6683
    Epoch 7/20
    63/63 [==============================] - 33s 516ms/step - loss: 0.5816 - accuracy: 0.6920 - val_loss: 0.5681 - val_accuracy: 0.7017
    Epoch 8/20
    63/63 [==============================] - 33s 515ms/step - loss: 0.5687 - accuracy: 0.6970 - val_loss: 0.5792 - val_accuracy: 0.7017
    Epoch 9/20
    63/63 [==============================] - 33s 517ms/step - loss: 0.5581 - accuracy: 0.7200 - val_loss: 0.5753 - val_accuracy: 0.6968
    Epoch 10/20
    63/63 [==============================] - 32s 511ms/step - loss: 0.5248 - accuracy: 0.7350 - val_loss: 0.5593 - val_accuracy: 0.7141
    Epoch 11/20
    63/63 [==============================] - 32s 507ms/step - loss: 0.5292 - accuracy: 0.7300 - val_loss: 0.5706 - val_accuracy: 0.7030
    Epoch 12/20
    63/63 [==============================] - 32s 513ms/step - loss: 0.5144 - accuracy: 0.7445 - val_loss: 0.5365 - val_accuracy: 0.7252
    Epoch 13/20
    63/63 [==============================] - 33s 518ms/step - loss: 0.5247 - accuracy: 0.7360 - val_loss: 0.5564 - val_accuracy: 0.7265
    Epoch 14/20
    63/63 [==============================] - 32s 508ms/step - loss: 0.5094 - accuracy: 0.7575 - val_loss: 0.5548 - val_accuracy: 0.7290
    Epoch 15/20
    63/63 [==============================] - 32s 509ms/step - loss: 0.5099 - accuracy: 0.7440 - val_loss: 0.5758 - val_accuracy: 0.7067
    Epoch 16/20
    63/63 [==============================] - 33s 513ms/step - loss: 0.4959 - accuracy: 0.7595 - val_loss: 0.5674 - val_accuracy: 0.7104
    Epoch 17/20
    63/63 [==============================] - 32s 511ms/step - loss: 0.5005 - accuracy: 0.7475 - val_loss: 0.6096 - val_accuracy: 0.6671
    Epoch 18/20
    63/63 [==============================] - 32s 508ms/step - loss: 0.4881 - accuracy: 0.7670 - val_loss: 0.5290 - val_accuracy: 0.7302
    Epoch 19/20
    63/63 [==============================] - 32s 511ms/step - loss: 0.4705 - accuracy: 0.7795 - val_loss: 0.5141 - val_accuracy: 0.7525
    Epoch 20/20
    63/63 [==============================] - 33s 515ms/step - loss: 0.4597 - accuracy: 0.7875 - val_loss: 0.5209 - val_accuracy: 0.7450



My model stabilized between 72-74% accuracy. This is better than the slightly better than model1. Considering the difference between training accuracy(~75%) and validation accuracy(~73%), it is clear that there was not much overfitting with this model.

Now we will visualize the results.


```python
plot_model(history3)
```

![](https://github.com/jameswest25/jameswest25.github.io/blob/master/images/HW3_Image_Classification_files/HW3_Image_Classification_15_0.png?raw=true)
    
## §5. Transfer Learning
So far, we’ve been training models for distinguishing between cats and dogs from scratch. In some cases, however, someone might already have trained a model that does a related task, and might have learned some relevant patterns. For example, folks train machine learning models for a variety of image recognition tasks. Maybe we could use a pre-existing model for our task?

To do this, we need to first access a pre-existing “base model”, incorporate it into a full model for our current task, and then train that model.

Paste the following code in order to download MobileNetV2 and configure it as a layer that can be included in our model.



```python
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

i = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = tf.keras.Model(inputs = [i], outputs = [x])
```
Now, we will create the model.

```python
model4 = tf.keras.Sequential([
    preprocessor,
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    base_model_layer,
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
model4.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history4 = model4.fit(train_dataset, 
                     epochs=20, 
                     validation_data=validation_dataset)
```

    Epoch 1/20
    63/63 [==============================] - 38s 557ms/step - loss: 0.7410 - accuracy: 0.8960 - val_loss: 0.0707 - val_accuracy: 0.9777
    Epoch 2/20
    63/63 [==============================] - 34s 531ms/step - loss: 0.1442 - accuracy: 0.9400 - val_loss: 0.0525 - val_accuracy: 0.9777
    Epoch 3/20
    63/63 [==============================] - 34s 537ms/step - loss: 0.1246 - accuracy: 0.9510 - val_loss: 0.0549 - val_accuracy: 0.9814
    Epoch 4/20
    63/63 [==============================] - 33s 529ms/step - loss: 0.1328 - accuracy: 0.9415 - val_loss: 0.0552 - val_accuracy: 0.9790
    Epoch 5/20
    63/63 [==============================] - 33s 529ms/step - loss: 0.1010 - accuracy: 0.9645 - val_loss: 0.0600 - val_accuracy: 0.9802
    Epoch 6/20
    63/63 [==============================] - 34s 533ms/step - loss: 0.0855 - accuracy: 0.9640 - val_loss: 0.0546 - val_accuracy: 0.9777
    Epoch 7/20
    63/63 [==============================] - 33s 528ms/step - loss: 0.0803 - accuracy: 0.9690 - val_loss: 0.0505 - val_accuracy: 0.9814
    Epoch 8/20
    63/63 [==============================] - 33s 529ms/step - loss: 0.0815 - accuracy: 0.9695 - val_loss: 0.0546 - val_accuracy: 0.9752
    Epoch 9/20
    63/63 [==============================] - 34s 532ms/step - loss: 0.0738 - accuracy: 0.9695 - val_loss: 0.0525 - val_accuracy: 0.9790
    Epoch 10/20
    63/63 [==============================] - 34s 534ms/step - loss: 0.0833 - accuracy: 0.9640 - val_loss: 0.0526 - val_accuracy: 0.9777
    Epoch 11/20
    63/63 [==============================] - 33s 526ms/step - loss: 0.0645 - accuracy: 0.9730 - val_loss: 0.0403 - val_accuracy: 0.9864
    Epoch 12/20
    63/63 [==============================] - 34s 532ms/step - loss: 0.0670 - accuracy: 0.9715 - val_loss: 0.0561 - val_accuracy: 0.9777
    Epoch 13/20
    63/63 [==============================] - 34s 532ms/step - loss: 0.0609 - accuracy: 0.9740 - val_loss: 0.0512 - val_accuracy: 0.9814
    Epoch 14/20
    63/63 [==============================] - 33s 527ms/step - loss: 0.0689 - accuracy: 0.9755 - val_loss: 0.0527 - val_accuracy: 0.9839
    Epoch 15/20
    63/63 [==============================] - 33s 528ms/step - loss: 0.0638 - accuracy: 0.9740 - val_loss: 0.0477 - val_accuracy: 0.9864
    Epoch 16/20
    63/63 [==============================] - 34s 534ms/step - loss: 0.0644 - accuracy: 0.9735 - val_loss: 0.0320 - val_accuracy: 0.9864
    Epoch 17/20
    63/63 [==============================] - 33s 530ms/step - loss: 0.0520 - accuracy: 0.9800 - val_loss: 0.0386 - val_accuracy: 0.9851
    Epoch 18/20
    63/63 [==============================] - 33s 529ms/step - loss: 0.0446 - accuracy: 0.9845 - val_loss: 0.0505 - val_accuracy: 0.9802
    Epoch 19/20
    63/63 [==============================] - 33s 529ms/step - loss: 0.0379 - accuracy: 0.9840 - val_loss: 0.0401 - val_accuracy: 0.9827
    Epoch 20/20
    63/63 [==============================] - 33s 528ms/step - loss: 0.0582 - accuracy: 0.9790 - val_loss: 0.0549 - val_accuracy: 0.9752



My model stabilized between 97-99% accuracy. This is much better accuracy than model1. Considering the difference between training accuracy(~98%) and validation accuracy(~98%), it is clear that there was little overfitting with this model

Now, we will plot the results.


```python
plot_model(history4)
```


    
![](https://github.com/jameswest25/jameswest25.github.io/blob/master/images/HW3_Image_Classification_files/HW3_Image_Classification_19_0.png?raw=true)
    

## §6. Score on Test Data
Finally, we will evaluate the accuracy of our most performant model on the unseen test_dataset.

```python
model4.evaluate(test_dataset)
```

    6/6 [==============================] - 3s 395ms/step - loss: 0.0776 - accuracy: 0.9583





    [0.0775797963142395, 0.9583333134651184]

Now, you have created an accurate Image Classification model. Congrats!
