
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
# This code has changed. Now instead of the ImageGenerator just rescaling
# the image, we also rotate and do other operations
# Updated to do image augmentation
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
# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR, # This is the source directory for training images
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary',
    # All images will be resized to 150x150
    target_size=(150,150)
) 

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)
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

Of course, image augmentation isn't the magic bullet to cure overfitting. It really helps to have a massive diversity of images. So for example, if we look at the horses or humans data set and train it for the same epochs, then we can take a look at its behavior. So I'm going to start training and show all 100 epochs. As you watch, you'll see the test accuracy climbing steadily. At first, the validation accuracy seems to be in step, but then you'll see it varying wildly. What's happening here is that despite the image augmentation, the diversity of images is still too sparse and the validation set may also be poorly designed, namely that the type of image in it is too close to the images in the training set. If you expect the data for yourself you'll see that's the case. For example, the humans are almost always standing up and in the center of the picture, in both the training and validation sets, so augmenting the image will change it to look like something that doesn't look like what's in the validation set. So by the time the training has completed, we can see the same pattern. The training accuracy is trending towards 100 percent, but the validation is fluctuating in the 60s and 70s. Let's plot this, we can see that the training accuracy climbs steadily in the way that we would want, but the validation fluctuated like crazy. So what we can learn from this is that the image augmentation introduces a random element to the training images but if the validation set doesn't have the same randomness, then its results can fluctuate like this. So bear in mind that you don't just need a broad set of images for training, you also need them for testing or the image augmentation won't help you very much.

There are a couple reasons behind such a scenario:
1. The training dataset is still too sparse 
2. The validation dataset is poorly designed in that it is very much similar to training set. Consider those two dataset is of high similarity, with IA, it extends the training dataset but those extended images are not included in validation set, making IA of less value. 

To summarize, IA introduces randomness to training set but cannot do the same to validation set. 
If the validation set lacks randomness, then it shows fluctuation. So it is important to make sure that both your training and validation sets contain a variety of random images, or IA may not be of much help.   

  