# CNN-Tensorflow



## WEEK-1 CNN

Look at a famous Kaggle Dogs vs Cats dataset: https://www.kaggle.com/c/dogs-vs-cats

### WEEK-1 Quiz

`flow_from_directory` on the ImageGenerator automatically labels images based on their directory name.

An image of size `150 * 150` passed into a `3 * 3` Convolution, the resulting size is `148 * 148` ((p-f+1) * (p-f+1)).

An image of size `150*150` used Pooling of size `2 * 2`, the resulting image is like `75 * 75`. 

If you want to view the history of your training, create a variable `history` and assign it to the return of `model.fit`or `model.fit_generator`.

`model.layers` API allows you to inspect the impact of convolutions on the image.

The reason why overfitting is more likely to occur on smaller datasets is that there's less likelihood of ALL possible features being encountered in the training process.


### Build CNN Model

```python
import tensorflow as tf

# define a Sequential layer as DNN, adding some Convolutional layer first with input_shape 150*150
# add a couple of convolutional layers, and flatten the final result to feed into the densely connected layers
# end our network with a sigmoid activation

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #Flatten the results to feed into DNN
    tf.keras.layers.Flatten(),
    #512 neurons in hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Configure the CNN Model
Train the model with `binary_crossentropy` loss, use `RMSprop` optimizer with a learning rate of 0.001 and monitor classification `accuracy`

```python
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'] )
``` 

#### RMSprop
#### SGD
stochastic gradient descent
#### ADAM

 
### Data Preprocessing
Create two data generators for both training and testing images. Normalize the image values to be in the [0, 1] range.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0/255)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=20, class_mode='binary', target_size=(150,150))

# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size=20, class_mode='binary', target_size=(150,150))
```

### Training

```python
history = model.fit_generator(train_generator, 
                                validation_data=test_generator, 
                                steps_per_epoch=100, 
                                epochs=15, 
                                validation_step=50, 
                                verbose=2)
```

### Visualizing Intermediate Representations

Because these representations carry increasingly less information about the original pixels of the image, 
but increasingly refined information about the class of the image. In a word, Convnet can be seen as a `DISTILLATION` process.
 
### Evaluating Accuracy and Loss from model history
```python
import matplotlib.pyplot as plt

train_acc = history.history['acc']
validation_acc = history.history['val_acc']
train_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs = range(len(train_acc))
#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, train_acc)
plt.plot(epochs, validation_acc)
plt.title("Training and Validation Accuracy")
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, train_loss)
plt.plot(epochs, validation_loss)
plt.title("Training and Validation Loss")
plt.figure()
```
Overfitting occurs as training accuracy gets close to 100% while testing accuracy stalls as 70%. The testing loss reaches the minimum after 5 epochs. 

## WEEK-2 Image Augmentation 

### WEEK-2 Quiz

Using IA effectively stimulates having a larger dataset for training.

Using IA makes your training gets slower, because the augmented data is bigger.

IA helps solve overfitting by manipulating the training dataset by generating more scenarios for features in the images.

IA does the image modification in memory and does no harm to your raw data on the disk. 

To use IA in Tensorflow, you simply add a few parameters to `ImageDataGenerator`.

`fill_mode` parameter makes use of attempts to recreate `lost information` after a transformation like a shear  



### What makes an Overfitting

For example, if the training accuracy gets close to 100% while validation accuracy is around 70% - 80%, then it is a typical overfitting problem. 

 
### Image Augmentation

It is a cheap way of `EXTENDING` your datasets by shifting, rotating, flipping, squash and zoom in/out. The line of reasoning is that you can mimic a lying cat by rotating 90 degrees of a standing cat in case 
your training dataset does not have a lying cat causing your model having difficulty identifying a lying one in validation process.


To add image augmentation, you simply add few lines of codes to `ImageDataGenerator`
```python
TRAINING_DIR='/tmp/cats-v-dogs/training/'

# this ImageDataGenerator allows you to instantiate generators of augmented image batches via .flow_from_directory
train_datagen = ImageDataGenerator(
                rescale=1.0/255,
                rotation_range=40, # values in degrees(0-180)
                width_shift_range=0.2, #fraction of total width or height within which to randomly shift
                height_shift_range=0.2,
                shear_rang=0.2,
                zoom_range=0.2,
                horizontao_flip=True,# flipping half of the images horizontally
                fill_mode='nearest' # strategy for filling in newly generated pixels most likely after rotation or shifting
                )

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=20,
    class_mode='binary',
    target_size=(150,150)
) 
```

#### use of batch_size

The `batch_size` is the amount of samples you feed into your network once. Say you have 1050 training samples and `batch_size=100`. It will take 
100 samples each time and train the network. Finally the last set of 50 samples will be a problem and a solution is to take out the last 50 and train it.

>Advantages of using a batch size < number of all samples 
>> 1. It requires less memory 
>> 1. Typically networks train faster with mini-batch

>Disadvantages of using a batch size < number of all samples
>> The less the batch the less the accurate estimate of the gradient will be

### Impact of Image Augmentation on Dogs vs. Cats CNN

With a simple code change, it turns out to have a better validation accuracy, improving overfitting issue in this case. 

### Impact of Image Augmentation on Horses vs. Human CNN

Under such circumstance, IA does not work well in that training accuracy is getting close to 100% while validation accuracy are fluctuating crazy in thee 60 to 70.

There are a couple reasons behind such a scenario:
1. The training dataset is still too sparse 
2. The validation dataset is poorly designed in that it is very much similar to training set. Consider those two dataset is of high similarity, with IA, it extends the training dataset but those 
extended images are not included in validation set, making IA of less value. To summarize, IA introduces randomness to training set but cannot do the same to validation set. 
If the validation set lacks randomness, then it shows fluctuation. So it is important to make sure that both your training and validation sets contain a variety of random images, or IA may not be of much help.   

  