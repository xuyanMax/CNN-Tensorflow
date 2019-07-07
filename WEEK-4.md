# WEEK-4

## Coding Summary 

## Data Preprocessing

### Download Images
```python
# you will train a CNN on the FULL Cats-v-dogs dataset
# This will require you doing a lot of data preprocessing because
# the dataset isn't split into training and validation for you
# This code block has all the required inputs
import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

!wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
    -O "/tmp/cats-and-dogs.zip"

## Rock-Paper-Scissor Data 
#!wget --no-check-certificate \
#   https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip \
#   -O /tmp/rps.zip
  
#!wget --no-check-certificate \
#    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip \
#    -O /tmp/rps-test-set.zip   
```
### Unzip file

```python
local_zip = '/tmp/cats-and-dogs.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
```

### Create Directories for Training and Validation

```python
# Use os.mkdir to create your directories
# You will need a directory for cats-v-dogs, and subdirectories for training
# and testing. These in turn will need subdirectories for 'cats' and 'dogs'
try:
    os.mkdir('/tmp/cats-v-dogs');
    os.mkdir('/tmp/cats-v-dogs/training');
    os.mkdir('/tmp/cats-v-dogs/validation');
    os.mkdir('/tmp/cats-v-dogs/validation/cats');
    os.mkdir('/tmp/cats-v-dogs/validation/dogs');
    os.mkdir('/tmp/cats-v-dogs/training/cats');
    os.mkdir('/tmp/cats-v-dogs/training/dogs');
except OSError:
    pass
```

### Split Images into Separate Directories 
```python
# Write a python function called split_data which takes
# a SOURCE directory containing the files
# a TRAINING directory that a portion of the files will be copied to
# a TESTING directory that a portion of the files will be copie to
# a SPLIT SIZE to determine the portion
# The files should also be randomized, so that the training set is a random
# X% of the files, and the test set is the remaining files
# SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
# Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
# and 10% of the images will be copied to the TESTING dir
# Also -- All images should be checked, and if they have a zero file length,
# they will not be copied over
#
# os.listdir(DIRECTORY) gives you a listing of the contents of that directory
# os.path.getsize(PATH) gives you the size of the file
# copyfile(source, destination) copies a file from source to destination
# random.sample(list, len(list)) shuffles a list
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
# YOUR CODE STARTS HERE
# YOUR CODE ENDS HERE
  files = []
  
  for filename in os.listdir(SOURCE):
    file = SOURCE + filename
    if os.path.getsize(file) > 0:
      files.append(filename)
    
  training_len = int(SPLIT_SIZE * len(files))
  validation_len = len(files) - training_len

  # randomize the source data
  rand_files = random.sample(files, len(files))
  training_set = rand_files[:training_len]
  validation_set = rand_files[-validation_len:]
  
  for filename in training_set:
      copyfile(SOURCE+filename, TRAINING+filename)
      
  for filename in validation_set:
      copyfile(SOURCE+filename, TESTING+filename)

CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"

TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/validation/cats/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/validation/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
```
### (1) Load Data from Direcories with Specified Naming

Create `XXX_datagen = ImageDataGenerator(...)` with `Image Augmentation` parameters (rotation, horizontal flipping, shear, shift, zoom and fill mode) for training and testing purposes. 

Load the images from dirctories `flow_from_directory`. It will automatically loads and categorizes the data according to directory naming.  
```python
XXX_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary' # multi-class use categorical
    )

```

### (2) Load Data from CSV Files Containing Labels and Pixels
You will need to write code that will read the file passed into this function. The first line contains the column headers, so you should ignore it. Each successive line contians 785 comma separated values between 0 and 255. 

The first value is the label. The rest are the pixel values for that picture
```python

def get_data(filename):
  # The function will return 2 np.array types. One with all the labels
  # One with all the images
  #
  # Tips: 
  # If you read a full line (as 'row') then row[0] has the label
  # and row[1:785] has the 784 pixel values
  # Take a look at np.array_split to turn the 784 pixels into 28x28
  # You are reading in strings, but need the values to be floats
  # Check out np.array().astype for a conversion
    with open(filename) as training_file:
      # Your code starts here
      tmp_images=[]
      tmp_labels=[]
      first_line = True
      csv_reader = csv.reader(training_file, deliminator=',')
      for row in csv_reader:
        if first_line:
          first_line=False
        else:
          image_data = row[1:785]
          image_label = row[0]
          image_data_np_array = np.array_split(image_data, 28)
          tmp_labels.append(image_label)
          tmp_images.append(image_data_np_array)
      images = np.array(tmp_images).astype('float')
      labels = np.array(tmp_labels).astype('float')
      # Your code ends here
    return images, labels

## Load 
training_images, training_labels = get_data('/sign_mnist_train.csv')
testing_images, testing_labels = get_data('/sign_mnist_test.csv')

## Add another dimension
training_images = np.expand_dims(training_images, 3)
testing_images = np.expand_dims(testing_images, 3)

## Create ImageDataGenerator & do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    filling_mode='nearest'
    )
validation_datagen = ImageDataGenerator(rescale=1./255.0)

test_generator = train_datagen.flow(training_images, training_labels)
validation_generator = validation_datagen.flow(testing_images, testing_labels)
## Define the Model: with no more than 2 Conv2D and 2 MaxPooling2D

model.tf.keras.models.Sequential([...])
## Configure, Compile the Model
model.compile()

## Train the model
history = model.fit_generator(...)
##

```

### OS Commands
`os.listdir(DIR)`: gives you a listing of the contents of the directory

`os.path.getsize(PATH)`: gives you the size of the file

`os.copyfile(SOURCE, DESTINATION)`: copies a file (not directory) from SOURCE to DESTINATION

`random.sample(list, len(list))`: shuffles a list

`os.path.join(SOURCE, filename)`: concatenates the path and filename giving a absolute path

`os.mkdir(PATH)`: creates your directory

### Numpy Commands

`np.array_split`

`np.array().astype`

`np.expand_dims`


## Visualizing Intermediate Images

## Model Compile Parameters

### Optimizers
`sgd`, `RMSprop`, `AdamOptimizer`

### loss
`mean_squared_error`, `sparse_categorical_crossentropy`, `binary_crossentropy`


