# MNIST Handwritten Digit Recognition with PyTorch

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset. The goal of this assignment was to achieve specific performance targets and implement certain architectural components.

## Assignment Objectives

The primary objectives for this model were:

*   **99.4% validation/test accuracy** (50/10k split)
*   **Less than 20,000 Parameters**
*   **Less than 20 Epochs**
*   Must use **Batch Normalization** and **Dropout**.
*   *(Optional)*: Use a **Fully Connected Layer** or **GAP** (Global Average Pooling).

## Model Architecture and Training Summary

The `MNISTModel` is a CNN designed with the following layers:

*   Two convolutional layers (`conv1`, `conv2`) followed by Batch Normalization and ReLU activation.
*   A Max Pooling layer (`torch.max_pool2d`) after `conv2`.
*   Another convolutional layer (`conv3`) followed by Batch Normalization and ReLU activation.
*   A 1x1 convolution layer (`conv1x1_last`) followed by ReLU, effectively acting as a channel reduction mechanism.
*   The output is then flattened and passed through two fully connected layers (`fc1`, `fc2`) for classification.

**Key Results:**

*   **Total Trainable Parameters:** 10183
    *   *Confirmation of < 20k Parameters: YES*
*   **Batch Normalization:** Used in `conv1`, `conv2`, and `conv3` layers.
    *   *Confirmation of use of Batch Normalization: YES*
*   **Dropout:** The current model implementation **does not explicitly include Dropout layers**. This would need to be added to meet the assignment requirement.
    *   *Confirmation of use of Dropout: NO (needs implementation)*
*   **Fully Connected Layer or GAP:** A 1x1 convolution (similar to Global Average Pooling for channel reduction) is used, followed by fully connected layers.
    *   *Confirmation of use of a Fully Connected Layer or GAP: YES*
*   **Validation/Test Accuracy:**
    *   Epoch 1 Test Accuracy: 96.20%
    *   Epoch 2 Test Accuracy: 97.16%
    *   *Target Accuracy (99.4%) was not met with the provided training output.*
*   **Epochs:** The training loop is configured for 20 epochs, but the provided output shows results only up to Epoch 2. The loop has a `break` condition if `epoch > 0 and epoch % 15 == 0`, which means it would train for at most 15 epochs if the condition is met.

## Image Preprocessing: Cropping Analysis

An initial exploration was conducted to see if MNIST images could be cropped to reduce their spatial dimensions by removing "black pixel padding."

*   **Method:** A `crop_mnist_images_cuda_optimized` function was implemented to find a global bounding box for non-black pixels across all images.
*   **Result:** The analysis indicated that no significant cropping was possible without losing information. The optimal crop boundaries were found to be `[0:28, 0:28]`, meaning the entire 28x28 image area contained relevant pixels in at least some samples.
*   **Boundary Touching Images:**
    *   Min row (0): 2 images had pixels in the top-most row.
    *   Max row (27): 812 images had pixels in the bottom-most row.
    *   Min col (0): 16 images had pixels in the left-most column.
    *   Max col (27): 92 images had pixels in the right-most column.

## Feature Map Visualization and Statistics (Example: Digit 7)

The notebook includes a detailed analysis of feature maps for a sample digit (digit 7) to understand how features are learned and transformed through the network.

**Statistics for Digit 7:**

| Layer             | Shape              | Mean Activation | Std Activation | Min Activation | Max Activation | Sparsity | Size/Channel Reduction |
| :---------------- | :----------------- | :-------------- | :------------- | :------------- | :------------- | :------- | :--------------------- |
| CONV1 (BN+ReLU)   | `[10, 8, 28, 28]`  | 0.3153          | 0.5941         | 0.0000         | 6.1359         | 57.89%   | -                      |
| CONV2 (BN+ReLU)   | `[10, 16, 28, 28]` | 0.2930          | 0.4687         | 0.0000         | 5.3045         | 41.95%   | -                      |
| MAXPOOL1 (2x2)    | `[10, 16, 14, 14]` | 0.4391          | 0.6112         | 0.0000         | 5.3045         | 29.78%   | 75.00% (from Conv2)    |
| CONV3 (BN+ReLU)   | `[10, 32, 14, 14]` | 0.4969          | 0.7603         | 0.0000         | 6.8159         | 49.07%   | -                      |
| 1x1 CONV (ReLU)   | `[10, 1, 14, 14]`  | 2.1741          | 2.7361         | 0.0000         | 19.2995        | 37.09%   | 96.88% (from Conv3)    |

*   **Observation:** Max Pooling significantly reduces spatial dimensions (75% reduction from Conv2 output), while the 1x1 convolution effectively reduces the number of channels (96.88% reduction from Conv3 output). ReLU activations introduce sparsity (zero activations), which varies across layers.

## Usage

To reproduce the training and visualization, run the `MNIST.ipynb` notebook. The visualization functions are called at the end of the notebook to display original images, feature maps, and detailed statistics for a chosen digit (defaulting to digit 7).

```python
# Assuming your model is trained and test_loader is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Visualize digit 7 (you can change this to any digit 0-9)
run_visualization_example(model, test_loader, digit=7)

# The visualization now includes:
# 1. Original images
# 2. Conv1 output (28x28, 8 channels)
# 3. Conv2 output (28x28, 16 channels)
# 4. MaxPool1 output (14x14, 16 channels)
# 5. Conv3 output (14x14, 32 channels)
# 6. 1x1 Conv output (14x14, 1 channel)

# Color coding for visualizations:
# - Blue titles/colormap = Convolutional layers
# - Red titles/colormap = Max pooling layers
# - Green titles/colormap = 1x1 Convolution layer (used for channel reduction/GAP)
```

**Note:** The model currently does not meet the 99.4% accuracy or the Dropout requirement. Further tuning and modifications to the model architecture and training regimen would be necessary to achieve these targets.

## Full Notebook Execution Outputs

### Cell 4 Output

```
torch.Size([70000, 28, 28])
```

### Cell 5 Output

```
Original shape: torch.Size([70000, 28, 28])
Cropped shape: torch.Size([70000, 28, 28])
Crop boundaries (min_row, max_row, min_col, max_col): (0, 28, 0, 28)
Reduction: 784 -> 784 pixels per image
Images touching boundaries:
Min row (0): 2 images
Max row (27): 812 images
Min col (0): 16 images
Max col (27): 92 images

First few image indices for each boundary:
min_row: [12352, 12905]
max_row: [71, 91, 123, 196, 282]
min_col: [1322, 11603, 11898, 15513, 21699]
max_col: [615, 627, 1243, 1852, 2893]

==================================================
SHOWING SPECIFIC BOUNDARY IMAGES
==================================================

Detailed Analysis:
Image 12352 (touches top):
  - Label: 2
  - Max pixel value in top row (0): 0.847
  - Number of pixels > 0.01 in top row: 4
Image 71 (touches bottom):
  - Label: 7
  - Max pixel value in bottom row (27): 0.902
  - Number of pixels > 0.01 in bottom row: 3

Crop boundaries determined by these extreme images:
Final crop will be [0:28, 0:28]
This removes 0 rows/cols of pure black pixels
```

### Cell 8 Output

```
MNISTDataset(data=60000)
MNISTDataset(data=10000)
Using device: cpu

Model Parameter Breakdown:
Layer: conv1.weight         | Parameters: 72
Layer: conv1.bias           | Parameters: 8
Layer: bn1.weight           | Parameters: 8
Layer: bn1.bias             | Parameters: 8
Layer: conv2.weight         | Parameters: 1152
Layer: conv2.bias           | Parameters: 16
Layer: bn2.weight           | Parameters: 16
Layer: bn2.bias             | Parameters: 16
Layer: conv3.weight         | Parameters: 4608
Layer: conv3.bias           | Parameters: 32
Layer: bn3.weight           | Parameters: 32
Layer: bn3.bias             | Parameters: 32
Layer: conv1x1_last.weight  | Parameters: 32
Layer: conv1x1_last.bias    | Parameters: 1
Layer: fc1.weight           | Parameters: 3920
Layer: fc1.bias             | Parameters: 20
Layer: fc2.weight           | Parameters: 200
Layer: fc2.bias             | Parameters: 10

Total Trainable Parameters: 10183
Epoch 1, Batch 0, Records 0, Loss: 2.347515344619751
Epoch 1, Batch 1, Records 10000, Loss: 0.3358577489852905
Epoch 1, Batch 2, Records 20000, Loss: 0.2632468044757843
Epoch 1, Batch 3, Records 30000, Loss: 0.2923506498336792
Epoch 1, Batch 4, Records 40000, Loss: 0.14531531929969788
Epoch 1, Batch 5, Records 50000, Loss: 0.12939102947711945
Accuracy on test data: 96.20%
Epoch 2, Batch 0, Records 0, Loss: 0.16280217468738556
Epoch 2, Batch 1, Records 10000, Loss: 0.1284981220960617
Epoch 2, Batch 2, Records 20000, Loss: 0.09894248098134995
Epoch 2, Batch 3, Records 30000, Loss: 0.08866942673921585
Epoch 2, Batch 4, Records 40000, Loss: 0.1541154831647873
Epoch 2, Batch 5, Records 50000, Loss: 0.09940677881240845
Accuracy on test data: 97.16%
Epoch 3, Batch 0, Records 0, Loss: 0.06052285060286522
Epoch 3, Batch 1, Records 10000, Loss: 0.17183956503868103
Epoch 3, Batch 2, Records 20000, Loss: 0.02694789692759514
Epoch 3, Batch 3, Records 30000, Loss: 0.056227907538414
Epoch 3, Batch 4, Records 40000, Loss: 0.14283299446105957
Epoch 3, Batch 5, Records 50000, Loss: 0.07166014015674591
Accuracy on test data: 97.35%
Epoch 4, Batch 0, Records 0, Loss: 0.05615822970867157
Epoch 4, Batch 1, Records 10000, Loss: 0.07470405846834183
Epoch 4, Batch 2, Records 20000, Loss: 0.0694851279258728
Epoch 4, Batch 3, Records 30000, Loss: 0.034334052354097366
Epoch 4, Batch 4, Records 40000, Loss: 0.08257039606571198
Epoch 4, Batch 5, Records 50000, Loss: 0.038479700684547424
Accuracy on test data: 97.80%
Epoch 5, Batch 0, Records 0, Loss: 0.07406769692897797
Epoch 5, Batch 1, Records 10000, Loss: 0.053800638765096664
Epoch 5, Batch 2, Records 20000, Loss: 0.04873738810420036
Epoch 5, Batch 3, Records 30000, Loss: 0.07632628083229065
Epoch 5, Batch 4, Records 40000, Loss: 0.02705494314432144
Epoch 5, Batch 5, Records 50000, Loss: 0.06316279619932175
Accuracy on test data: 97.97%
Epoch 6, Batch 0, Records 0, Loss: 0.0357833057641983
Epoch 6, Batch 1, Records 10000, Loss: 0.06830509752035141
Epoch 6, Batch 2, Records 20000, Loss: 0.07684725522994995
Epoch 6, Batch 3, Records 30000, Loss: 0.03365922346711159
Epoch 6, Batch 4, Records 40000, Loss: 0.030502755203843117
Epoch 6, Batch 5, Records 50000, Loss: 0.034942738711833954
Accuracy on test data: 98.14%
Epoch 7, Batch 0, Records 0, Loss: 0.026744049787521362
Epoch 7, Batch 1, Records 10000, Loss: 0.02773121550679207
Epoch 7, Batch 2, Records 20000, Loss: 0.01358602661639452
Epoch 7, Batch 3, Records 30000, Loss: 0.03719940781593323
Epoch 7, Batch 4, Records 40000, Loss: 0.0519302636384964
Epoch 7, Batch 5, Records 50000, Loss: 0.06173255294561386
Accuracy on test data: 98.41%
Epoch 8, Batch 0, Records 0, Loss: 0.009971988387405872
Epoch 8, Batch 1, Records 10000, Loss: 0.04169600009918213
Epoch 8, Batch 2, Records 20000, Loss: 0.02081700600683689
Epoch 8, Batch 3, Records 30000, Loss: 0.05151614919304848
Epoch 8, Batch 4, Records 40000, Loss: 0.02293025702238083
Epoch 8, Batch 5, Records 50000, Loss: 0.04561937976725
Accuracy on test data: 98.40%
Epoch 9, Batch 0, Records 0, Loss: 0.047806501388549805
Epoch 9, Batch 1, Records 10000, Loss: 0.024845495819076896
Epoch 9, Batch 2, Records 20000, Loss: 0.028911571949720383
Epoch 9, Batch 3, Records 30000, Loss: 0.021008681505918503
Epoch 9, Batch 4, Records 40000, Loss: 0.043423790484666824
Epoch 9, Batch 5, Records 50000, Loss: 0.03192257881164551
Accuracy on test data: 98.60%
Epoch 10, Batch 0, Records 0, Loss: 0.017772749066352844
Epoch 10, Batch 1, Records 10000, Loss: 0.009745129197835922
Epoch 10, Batch 2, Records 20000, Loss: 0.03843513876199722
Epoch 10, Batch 3, Records 30000, Loss: 0.028678081929683685
Epoch 10, Batch 4, Records 40000, Loss: 0.027063079178333282
Epoch 10, Batch 5, Records 50000, Loss: 0.02083656715568574
Accuracy on test data: 98.74%
Epoch 11, Batch 0, Records 0, Loss: 0.0053912629672236
Epoch 11, Batch 1, Records 10000, Loss: 0.012574672773480415
Epoch 11, Batch 2, Records 20000, Loss: 0.03472098708152771
Epoch 11, Batch 3, Records 30000, Loss: 0.029851600527763367
Epoch 11, Batch 4, Records 40000, Loss: 0.026367347925901413
Epoch 11, Batch 5, Records 50000, Loss: 0.00760447161638662
Accuracy on test data: 98.78%
Epoch 12, Batch 0, Records 0, Loss: 0.009492167085409164
Epoch 12, Batch 1, Records 10000, Loss: 0.027202358469367027
Epoch 12, Batch 2, Records 20000, Loss: 0.007802103459835052
Epoch 12, Batch 3, Records 30000, Loss: 0.015707736015319824
Epoch 12, Batch 4, Records 40000, Loss: 0.008435777835547924
Epoch 12, Batch 5, Records 50000, Loss: 0.015848558396101
Accuracy on test data: 98.88%
Epoch 13, Batch 0, Records 0, Loss: 0.019176378846168518
Epoch 13, Batch 1, Records 10000, Loss: 0.005471627667173743
Epoch 13, Batch 2, Records 20000, Loss: 0.0028753238264471292
Epoch 13, Batch 3, Records 30000, Loss: 0.008545806631445885
Epoch 13, Batch 4, Records 40000, Loss: 0.005160868167877197
Epoch 13, Batch 5, Records 50000, Loss: 0.005470878146588802
Accuracy on test data: 98.82%
Epoch 14, Batch 0, Records 0, Loss: 0.0039237699918448925
Epoch 14, Batch 1, Records 10000, Loss: 0.01166311651468277
Epoch 14, Batch 2, Records 20000, Loss: 0.0051410915364169
Epoch 14, Batch 3, Records 30000, Loss: 0.011933066695928574
Epoch 14, Batch 4, Records 40000, Loss: 0.0063235256522893906
Epoch 14, Batch 5, Records 50000, Loss: 0.0021669472716748714
Accuracy on test data: 99.02%
Epoch 15, Batch 0, Records 0, Loss: 0.003848725836724043
Epoch 15, Batch 1, Records 10000, Loss: 0.0019234676472842693
Epoch 15, Batch 2, Records 20000, Loss: 0.006969567500054836
Epoch 15, Batch 3, Records 30000, Loss: 0.004410191837949364
Epoch 15, Batch 4, Records 40000, Loss: 0.016335198652029037
Epoch 15, Batch 5, Records 50000, Loss: 0.008719570650160313
Accuracy on test data: 98.92%
Epoch 16, Batch 0, Records 0, Loss: 0.0032900993525981903
Epoch 16, Batch 1, Records 10000, Loss: 0.0028169608107209206
Epoch 16, Batch 2, Records 20000, Loss: 0.0028886310756206512
Epoch 16, Batch 3, Records 30000, Loss: 0.0020739957317709923
Epoch 16, Batch 4, Records 40000, Loss: 0.00845348834991455
Epoch 16, Batch 5, Records 50000, Loss: 0.001150047382503748
Accuracy on test data: 99.09%
Epoch 17, Batch 0, Records 0, Loss: 0.002967675129696727
Epoch 17, Batch 1, Records 10000, Loss: 0.0012759972317102157
Epoch 17, Batch 2, Records 20000, Loss: 0.0009497745759785175
Epoch 17, Batch 3, Records 30000, Loss: 0.00078860714770481
Epoch 17, Batch 4, Records 40000, Loss: 0.0007954932609573007
Epoch 17, Batch 5, Records 50000, Loss: 0.0021237467043101788
Accuracy on test data: 99.08%
Epoch 18, Batch 0, Records 0, Loss: 0.0006764516718685627
Epoch 18, Batch 1, Records 10000, Loss: 0.0009714801740832329
Epoch 18, Batch 2, Records 20000, Loss: 0.0004945115149952471
Epoch 18, Batch 3, Records 30000, Loss: 0.000958189679836
Epoch 18, Batch 4, Records 40000, Loss: 0.0005706173070892692
Epoch 18, Batch 5, Records 50000, Loss: 0.0018861314933374524
Accuracy on test data: 99.14%
Epoch 19, Batch 0, Records 0, Loss: 0.00030564619962111115
Epoch 19, Batch 1, Records 10000, Loss: 0.000185960010037303
Epoch 19, Batch 2, Records 20000, Loss: 0.00010978643997618929
Epoch 19, Batch 3, Records 30000, Loss: 0.0001648792011058
Epoch 19, Batch 4, Records 40000, Loss: 0.00010041285227708519
Epoch 19, Batch 5, Records 50000, Loss: 0.00012574041724205017
Accuracy on test data: 99.19%
Epoch 20, Batch 0, Records 0, Loss: 0.00011566851608400047
Epoch 20, Batch 1, Records 10000, Loss: 0.00011218739217612893
Epoch 20, Batch 2, Records 20000, Loss: 8.875883398903564e-05
Epoch 20, Batch 3, Records 30000, Loss: 8.948270525224507e-05
Epoch 20, Batch 4, Records 40000, Loss: 9.996174783442914e-05
Epoch 20, Batch 5, Records 50000, Loss: 7.026756854727492e-05
Accuracy on test data: 99.19%
```

### Cell 11 Output

```
=== Analyzing Digit 7 ===
1. Showing original images...
2. Showing feature maps for each convolutional layer...
Getting 10 samples of digit 7...

Visualizing Conv1 (1→8 channels) - 28×28
Feature map shape: torch.Size([10, 8, 28, 28])

Visualizing Conv2 (8→16 channels) - 28×28
Feature map shape: torch.Size([10, 16, 28, 28])

Visualizing MaxPool1 (after Conv2) - 14×14
Feature map shape: torch.Size([10, 16, 14, 14])

Visualizing Conv3 (16→32 channels) - 14×14
Feature map shape: torch.Size([10, 32, 14, 14])

Visualizing 1x1 Conv (32→1 channels) - 14×14
Feature map shape: torch.Size([10, 1, 14, 14])
3. Analyzing feature statistics...

Feature Map Statistics for Digit 7:
------------------------------------------------------------
CONV1 (BN+ReLU):
  Shape: torch.Size([10, 8, 28, 28]) (expected: [10, 8, 28, 28])
  Mean activation:   0.3153
  Std activation:    0.5941
  Min activation:    0.0000
  Max activation:    6.1359
  Sparsity:           57.89%

CONV2 (BN+ReLU):
  Shape: torch.Size([10, 16, 28, 28]) (expected: [10, 16, 28, 28])
  Mean activation:   0.2930
  Std activation:    0.4687
  Min activation:    0.0000
  Max activation:    5.3045
  Sparsity:           41.95%

MAXPOOL1 (2x2):
  Shape: torch.Size([10, 16, 14, 14]) (expected: [10, 16, 14, 14])
  Mean activation:   0.4391
  Std activation:    0.6112
  Min activation:    0.0000
  Max activation:    5.3045
  Sparsity:           29.78%
  Size reduction:     75.00% (from Conv2)

CONV3 (BN+ReLU):
  Shape: torch.Size([10, 32, 14, 14]) (expected: [10, 32, 14, 14])
  Mean activation:   0.4969
  Std activation:    0.7603
  Min activation:    0.0000
  Max activation:    6.8159
  Sparsity:           49.07%

1x1 CONV (ReLU):
  Shape: torch.Size([10, 1, 14, 14]) (expected: [10, 1, 14, 14])
  Mean activation:   2.1741
  Std activation:    2.7361
  Min activation:    0.0000
  Max activation:   19.2995
  Sparsity:           37.09%
  Channel reduction:    96.88% (from Conv3)
```

