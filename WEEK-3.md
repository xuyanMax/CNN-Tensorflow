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

