# Projected Gradient Descent image inversion
## Model invertibility 
In the space of model robustness, understanding the representation space (penultimate layer vector space) of neural networks is a valuable metric. The smoothness of the representation space is an indicator of how easily an adversary can make an imperceptible change in the pixel space that dramatically impacts the loss of the classifier on that image.

This experiment is seeing if we can find a certain target image by traversing the image space and making steps via projected gradient descent evaluated by loss in the representation space.

## Setup
Here we use a few state-of-the-art CNN model architectures, primarily Resnet50. The robustness library https://github.com/MadryLab/robustness is used to make the "adversarial attacks" i.e. traverse the representation space to go from source to target image and try to match the target exactly.

inversion.py gets random images and vector representations. It then takes an "inversion loss" to check the vector similarity of the 2 representations. compare.py gets the mean pixel difference between 2 images; to check if the images match. run_restricted_imagenet_l2_eps30.py (or any of the run files) shows the process of taking a PGD step from the representation space, and calculating loss in the representation space.

## Plotting gradients/images
plot_optim.py plots the gradients at any point in the representation space - plotted in a sequence of representations it shows how smoothly the gradient is changing. plot_loss.py plots the representation loss through the process to show the progress in representation similarity as PGD progresses.
