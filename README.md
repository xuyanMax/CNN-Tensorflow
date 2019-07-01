# CNN-Tensorflow



## WEEK-1 

Look at a famous Kaggle Dogs vs Cats dataset: https://www.kaggle.com/c/dogs-vs-cats

### Some important skills learnt from it

`flow_from_directory` on the ImageGenerator automatically labels images based on their directory name.

An image of size `150 * 150` passed into a `3 * 3` Convolution, the resulting size is `148 * 148` ((p-f+1) * (p-f+1)).

An image of size `150*150` used Pooling of size `2 * 2`, the resulting image is like `75 * 75`. 

If you want to view the history of your training, create a variable `history` and assign it to the return of `model.fit`or `model.fit_generator`.

`model.layers` API allows you to inspect the impact of convolutions on the image.

The reason why overfitting is more likely to occur on smaller datasets is that there's less likelihood of ALL possible features being encountered in the training process.


### Build CNN Model

