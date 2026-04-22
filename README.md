# Sounds of Seattle Birds (Neural Networks)

A deep learning project that classifies Seattle-area bird sounds from spectrogram images using **convolutional neural networks (CNNs)**.

## Overview

This project explores bird sound classification using audio recordings from **twelve bird species commonly found in the Seattle region**. The recordings were converted into spectrograms so that sound patterns could be analyzed as images.

Two classification tasks were developed:

- **Binary classification**: distinguish **Song Sparrow** from **White-crowned Sparrow**
- **Multi-class classification**: identify **12 bird species** from spectrogram images

The goal of the project was to evaluate how well neural networks can recognize bird species from audio-based visual features.

## Dataset

The dataset consists of preprocessed **MP3 bird recordings** stored in a single **HDF5 file**, with each species organized into its own group.

### Data characteristics

- **12 bird species**
- Audio clips converted into **spectrogram images**
- Images resized to **128 × 517 pixels**
- Pixel values normalized between **0 and 1**

The spectrogram representation captures how frequency content changes over time, making it useful for distinguishing bird calls.

## Technical Background

This project focuses on **Convolutional Neural Networks (CNNs)**, which are neural networks designed for image-like data. Since spectrograms can be treated as images, CNNs are a strong fit for this classification problem.

### Why CNNs?

- They can automatically learn visual patterns from spectrograms
- They work well for image recognition problems
- They reduce the need for manual feature engineering

### Challenges

- Overfitting during training
- Class imbalance across bird species
- Similar-sounding bird calls that are hard to separate
- Background noise in recordings

## Methodology

### Data Preparation

The workflow included:

- loading bird recordings from an **HDF5 dataset**
- converting audio clips into **spectrograms**
- resizing and normalizing spectrogram images
- mapping bird species to integer labels
- splitting the data into training, validation, and test sets

For the final task, three unlabeled test audio clips were also converted into **Mel spectrograms** for prediction.

### Binary Model

The binary CNN was trained to classify:

- **Song Sparrow**
- **White-crowned Sparrow**

Model design included:

- two convolution layers
- pooling layers
- a small fully connected layer
- **sigmoid** output
- **Adam optimizer**
- **binary cross-entropy loss**

To reduce overfitting, **early stopping** and **model checkpointing** were added.

### Multi-Class Model

The multi-class CNN extended the binary design by adding:

- an additional convolution layer
- pooling after each convolution block
- a dense output layer with **softmax**

This model predicted across **12 bird species**.

## Results

### Binary Classification

The binary classifier achieved performance in the **mid-60% test accuracy range**, with the report highlighting an **AUC of about 0.68** after improving the training process with early stopping.

Key takeaway:

- the model could separate the two sparrow species reasonably well
- confusion remained because the two species have similar calls and similar spectrogram patterns

### Multi-Class Classification

The multi-class model achieved about **41% test accuracy** after training improvements.

Key findings:

- the model performed better on species with more training examples
- species with fewer examples were often misclassified
- the model showed bias toward common classes
- acoustically similar bird calls were difficult to distinguish

## Key Findings

- **Early stopping** helped reduce overfitting
- **Class imbalance** had a major effect on performance
- **House Sparrow** and other common classes were predicted more often than rare classes
- Similar bird calls created substantial confusion between classes
- CNNs were useful, but performance was limited by noisy data and overlapping sound patterns

## Discussion

This project shows both the strengths and limitations of neural networks for bird-song classification.

CNNs were effective because spectrograms behave like images, allowing the model to learn patterns such as:

- trills
- harmonics
- repeated sound structures over time and frequency

However, performance was affected by:

- imbalanced class distributions
- overlapping acoustic patterns
- faint or noisy calls
- difficulty distinguishing visually similar spectrograms

## Conclusion

This study demonstrates that neural networks can be used for **bird species identification from audio recordings**, but the classification task remains challenging.

The binary model performed better than the multi-class model, while the multi-class system showed the difficulty of scaling bird-song recognition across many species. The results suggest that better balance across classes, improved preprocessing, and additional model refinement could improve future performance.


## Acknowledgments

- Seattle University course materials for **Statistical Machine Learning-2**
- Bird audio preprocessing workflow using spectrograms and HDF5 storage
- References used in the report for deep learning, CNNs, RNNs, and loss functions
