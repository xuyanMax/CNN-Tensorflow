# WEEK-3 Transfer Learning 

## Take Away

`Transfer Learning` is helpful in that it allows the usage of features trained from large datasets you may not have access to.

`layer.trainable` is the approach to locking or freezing a layer from retraining.

Your number of classifications differs from the model. To take advatange of the Transfer Learning, add your `DNN` at the bottom of the network, specifying your output layer (number of classes) you want.

`Image Augmentation` can be used in combination with `Transfer Learning` in that new `DNN` layers are added at the bottom of your network.

`Drop out` rate being to high can be dangerous as the network would lose specialization to the effect that hardly could lifting the accuracy. 

## Transfer Learning 

The idea behind `Transfer Learning` is to consider someone else's models, far more complex than yours, trained with a lot data. 

Key words: Available Complex Models, Take it, Build on it

### Lockup 
Their models have convolutional layers and they're here intact with features that already been learnt. So you can `lock` them (lock any layers, you get it from `model.summary`) instead of retraining them on your data. Next, the way of using these models (features learnt from its datasets that you may not have access to) trained with a large datasets and used the convolutions that it learnt when classifying its data, is to lock all the convolutions (you don't have to and can lock any one you want) and then `retrain the Dense layer` from that model with your data.

It is likely to retrain the lower convolutions for its particularly specialized for its images. It takes some trail and error to find out a suitable combination.

## Drop Out

It is a `layer` in Keras.

The parameter is between `0 and 1` and it's the fraction of units to drop. Say drop out as 0.2, it is dropping out 20% of total neurons of the network. 

The idea behind Drop Out is that even with Transfer Learning, we can still end up with `overfitting`. Because layers in network can some times have similar weight and possibly impact each other, which leads to overfitting, being a risk of a big complex model. 

## Implementation

### imports

```python
imports os
imports tensorflow.keras from layers
from tensorflow.keras import Model

# Download a copy of the pretrained weights for inception neural network 
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
from tensorflow.keras.applications.inception_v3 import inceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150,150,3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights(local_weights_file)

# Lock up layers
for layer in pre_trained_model.layers:
    layer.trainbale = False
# Show model layers & get the last layer
pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
```
### Build  & Compile Model Based on Pretrained Model

```python
from tensorflow.keras.optimizers import RMSprop

# Flatten the output to 1 dimension
x = layers.Flatten()(last_output)
# add layers
x = layers.Dense(1024, activation='relu')
x = layers.Dropout(0,2)(x)
x = layers.Dense(1, activation='sigmoid')
# Build new model
model = Model(pre_trained_model, x)

model.compile(optimizer = 'RMSprop',
    loss='binary_crossentropy',
    metrics=['acc'])    
}
```

### Create ImageDataGenerators for training and validation 
```python

train_dir = '/tmp/cats_and_dogs_filtered/training'
train_cats_dir = '/tmp/cats_and_dogs_filtered/training/cats'
train_cats_dir = '/tmp/cats_and_dogs_filtered/training/dogs'

validation_dir = '/tmp/cats_and_dogs_filtered/validation'ython
validation_cats_dir = '/tmp/cats_and_dogs_filtered/validation/cats'
validation_dogs_dir = '/tmp/cats_and_dogs_filtered/validation/dogs'

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255.0, ...)
# Note that the validation data should not be augmented!
validation_datagen = ImageDataGenerator(rescale=1./255.0)
# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
```

### Create callbacks to stop training once accuracy reaches 99.9%
```python
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(log.get('acc') > 0.99):
            print("\nReached 99.9% accuracy to cancelling training.")
            self.model.stop_training=True
```
### Configure the Model

```python
history = model.fit_generator(train_generator, 
                            validation_data = validation_generator,
                            epochs = 100, 
                            steps_per_epoch = 50,
                            validation_steps = 50,
                            verbose = 2,
                            callbacks=[callbacks])
```

### Plot the Accuracy 

```python
import matplotlib.pyplot as Plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = len(acc)

plt.plot(epochs, acc, 'r', label="training acc")
plt.plot(epochs, val_acc, 'r', label="validation acc")
plt.title("Training and validation accuracy")
plt.legend(loc=0)
plt.figure()
plt.show()
```

