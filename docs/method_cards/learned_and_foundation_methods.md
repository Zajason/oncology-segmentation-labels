# Learned and Foundation Methods

These methods use pretrained or trained neural networks and require GPU acceleration for practical runtimes.

## SAM

Segment Anything is used as a bbox-prompted foundation model baseline. It requires a SAM checkpoint and can produce strong masks without task-specific training.

## U-Net

U-Net is the supervised medical segmentation baseline. It is trained with ground-truth masks and represents the strongest fully supervised mask-generation setting in this project.

## Guided U-Net

Guided U-Net receives both the image and a bbox-derived guidance channel. It directly models the box-to-mask task and is especially relevant to weak-label synthetic mask generation.

## Why include these methods?

Classical methods show what is possible without training. SAM tests a general foundation model. U-Net and Guided U-Net test how much performance improves when the method learns the dataset-specific mask distribution.
