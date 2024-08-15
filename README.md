# Human-in-the-Loop Segmentation of Multi-species Coral Imagery

The official repository for the paper: "Human-in-the-Loop Segmentation of Multi-species Coral Imagery".

If this repository contributes to your research, please consider citing the publication below.

```
Raine, S., Marchant, R., Kusy, B., Maire, F., Suenderhauf, N. and Fischer, T. 2024. Human-in-the-Loop Segmentation of Multi-species Coral Imagery.  Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2024.
```

### Bibtex
```
@inproceedings{raine2024human,
  title={Human-in-the-Loop Segmentation of Multi-species Coral Imagery},
  author={Raine, Scarlett and Marchant, Ross and Kusy, Brano and Maire, Frederic and Suenderhauf, Niko and Fischer, Tobias},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2024}
}
```

The full paper can be accessed at: \[[Paper](https://arxiv.org/abs/2404.09406)].
Watch our video explaining the approach here: \[[Video](https://www.youtube.com/watch?v=YBTUCECu3OM)].

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Acknowledgements](#acknowledgements)

<a name="installation"></a>
## Installation
We suggest using the Mamba (or Anaconda) package manager to install dependencies.

1. Follow instructions at \[[Denoising ViT](https://github.com/Jiawei-Yang/Denoising-ViT)].
2. Additionally install faiss-gpu, torchmetrics, tqdm:

```mamba install -c pytorch faiss-gpu```
```mamba install -c conda-forge torchmetrics tqdm```

<a name="quick-start"></a>
## Quick Start 

The provided scripts perform point label propagation.  We provide a script for random and grid spaced labeling, and another which uses our Human-in-the-Loop labeling regime. 

Ensure you have a folder with images and another folder with corresponding dense ground truth masks.  We use these ground truth masks to simulate both the sparse label setting and the human expert.

The script will save the augmented ground truth masks in the specified directory as .png images, where each value indicates the class at that pixel in the corresponding image.

Run the script using:

```python point_label_propagation_hil.py```
or
```python point_label_propagation.py```

You must provide the following arguments:
* '-i', '-image_path', type=str, help='the path to the images', required=True
* '-g', '-gt_path', type=str, help='the path to the provided labels', required=True

Optional paths for saving augmented ground truth as integer masks for training semantic segmentation model, and/or visualising as RGB masks
* '-p', '--save_path', type=str, help='the destination of your propagated labels'
* '-v', '--vis_path', type=str, help='the destination of the RGB visualised propagated labels'

If NOT using the human-in-the-loop labeling regime, use the following to change the functionality:
* '--random', action='store_true', dest='random', help='use this flag when you would like to use randomly distributed point labels, otherwise they will be grid-spaced'

The following are optional arguments for all approaches:
* '-n', '--num_total_pixels', type=int, default=25, help='the number of labeled points per image'
* '-k', '--knn', type=int, default=1, help='k value in KNN'

The following are optional hyperparameters for human-in-the-loop labeling regime:
* '-s', '--sigma', type=int, default=50 help='sigma value, controls smoothness of the distance mask'
* '-l', '--lambda', type=float, default=2.2, help='lambda value, controls weighting of the inverted similarity and distance masks'
* '-m', '--num_human_pixels', type=int, default=10, help='the number of human-labeled points per image for initializing the approach'

Example 1: This is for the UCSD Mosaics dataset, using the HIL labeling regime, saving augmented ground truth masks only, with k=1 and 50 total pixels per image.

```python point_label_propagation_hil.py -i "mosaics/test/images/" -g "mosaics/test/labels/" -p "out-masks" -n 50```

Example 2: This is for the UCSD Mosaics dataset, assuming grid-spaced labels, saving augmented ground truth masks and visualising RGB augmented ground truth masks, with k=1 and 100 total pixels per image.

```python point_label_propagation.py -i "mosaics/test/images/" -g "mosaics/test/labels/" -v "vis" -p "out-masks" -n 100```

The UCSD Mosaics dataset can be downloaded from the authors of the CoralSeg paper: \[[Dataset](https://sites.google.com/a/unizar.es/semanticseg/home)]
Note that in this work, we remove a number of images where the ground truth mask has been corrupted.  A list of these corrupted files is in CorruptedImages.txt. 

<a name="acknowledgements"></a>
## Acknowledgements
This work was done in collaboration between the QUT Centre for Robotics and CSIRO's Data61. 

