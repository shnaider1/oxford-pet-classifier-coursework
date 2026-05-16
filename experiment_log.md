# Experiment Log

## Baseline setup

I created the initial project structure and added a simple CNN model. The first version used normal RGB images only.

I also added basic training and testing scripts to check that the model could load the Oxford-IIIT Pet dataset, train for a few epochs, save a model file, and calculate test accuracy.

## Residual model improvement

I improved the baseline CNN by adding simple residual blocks. These blocks use skip connections so the model can reuse earlier features.

I also updated the training script to save checkpoints after each epoch. This makes it easier to compare different stages of training.

## Trimap preprocessing

I started using the official Oxford-IIIT Pet trimap data. The trimap gives pixel-level information about the pet, background, and border.

At this stage, I used the trimap to replace background pixels with a neutral grey value. The model still used a normal 3-channel RGB input.
