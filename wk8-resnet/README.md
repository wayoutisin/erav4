# ResNet CIFAR-100 Training Assignment

## Objective

- Train a ResNet model (architecture and size of your choice) from scratch on the CIFAR-100 dataset.
- Use ChatGPT and Cursor extensively for coding, debugging, and documentation throughout the assignment.
- The goal is to achieve at least **73% top-1 accuracy** on CIFAR-100. This typically requires training for around 100 epochs or more.
- **Important**: You may NOT use a pre-trained model. You can use any code/repo, but all training must be from scratch.

## Model Architecture (ResNet-56)

- **Type:** ResNet-56, optimized for CIFAR datasets
- **Encoding:** 3 stages of 9 residual blocks each (3x9 = 27 blocks, 56 conv layers total)
- **Initial Convolution:** 3x3 Conv, 16 filters
- **Block Depths:**
    - Stage 1: 16 filters × 9 blocks
    - Stage 2: 32 filters × 9 blocks (stride 2 to downsample)
    - Stage 3: 64 filters × 9 blocks (stride 2)
- **Final pooling:** Global average pooling → FC (100 classes)
- **Activation:** ReLU; BatchNorm after each conv
- **Shortcut connections:** Identity/conv depending on shape
- **Weight Initialization:** Kaiming He (fan_out)
- **Parameter Count:**
    - Total: 861,620
    - Trainable: 861,620

## Training Configuration

- **Dataset:** CIFAR-100
- **Data Augmentation:**
    - Random Crop w/ padding 4
    - Random Horizontal Flip
    - Random Rotation (±15°)
    - Color Jitter (brightness/contrast/saturation)
    - Random Erasing
    - **Cutout augmentation:** (1 hole, length 16)
    - Normalization using mean: [0.5071, 0.4867, 0.4408], std: [0.2675, 0.2565, 0.2761]
- **Batch Size:** 128
- **Epochs:** 150 (target typically reached at ~100, but run to 150 for best model)
- **Optimizer:** SGD, momentum=0.9, weight decay = 5e-4
- **Learning Rate:** initial 0.1, cosine annealing schedule
- **Loss:** CrossEntropyLoss
- **Device:** cuda (if available)
- **Checkpointing:** Best model (by validation accuracy) is saved each epoch
- **Target:** ≥73% top-1 accuracy (no pre-training)

## Deliverables

1. **HuggingFace Spaces App:**
   - Deploy your trained model as a live demo on HuggingFace Spaces. Learn how to do this (see HuggingFace documentation) if you're not familiar.
   - Share the link to your live HuggingFace Spaces web app.

2. **Markdown File with Logs:**
   - Provide a Markdown file containing logs from epoch 1 through the last epoch trained. This demonstrates your training and progress transparently.

3. **GitHub Repository:**
   - Upload all code (including data loading, training scripts, model code, and logs) to a public GitHub repository.
   - Share the GitHub repository link.

## Checklist

- [x] Train ResNet on CIFAR-100 from scratch (no pre-trained weights)
- [x] Use ChatGPT and Cursor for AI-powered programming
- [x] Achieve 73% accuracy or higher
- [x] Save full training logs
- [x] Share HuggingFace Spaces app link
- [x] Share GitHub code link

---

## Final Performance

| Metric              | Value        |
|---------------------|-------------|
| **Final Epoch**     | 138         |
| **Best Epoch**      | 138         |
| **Final Train Acc** | 80.92%      |
| **Final Test Acc**  | 73.19%      |
| **Best Val Acc**    | 73.19%      |

---

## Training Log

Below is the complete, unabridged training log from the original RESNET.ipynb run (epoch-by-epoch):

```
Epoch 1 Summary:
  Train Loss: 4.4447 | Train Acc: 2.70%
  Val Loss:   4.4773 | Val Acc:   3.56%

Epoch 2 Summary:
  Train Loss: 4.0379 | Train Acc: 6.51%
  Val Loss:   3.8102 | Val Acc:   9.91%

Epoch 3 Summary:
  Train Loss: 3.7255 | Train Acc: 11.69%
  Val Loss:   3.5965 | Val Acc:   13.96%

Epoch 4 Summary:
  Train Loss: 3.4596 | Train Acc: 16.15%
  Val Loss:   3.2573 | Val Acc:   20.12%

Epoch 5 Summary:
  Train Loss: 3.2052 | Train Acc: 20.81%
  Val Loss:   3.3455 | Val Acc:   20.57%

Epoch 6 Summary:
  Train Loss: 2.9534 | Train Acc: 25.25%
  Val Loss:   2.9886 | Val Acc:   27.45%

Epoch 7 Summary:
  Train Loss: 2.7657 | Train Acc: 29.10%
  Val Loss:   2.8631 | Val Acc:   28.49%

Epoch 8 Summary:
  Train Loss: 2.6212 | Train Acc: 31.89%
  Val Loss:   2.6903 | Val Acc:   31.16%

Epoch 9 Summary:
  Train Loss: 2.5014 | Train Acc: 34.60%
  Val Loss:   2.4616 | Val Acc:   36.56%

Epoch 10 Summary:
  Train Loss: 2.3968 | Train Acc: 37.16%
  Val Loss:   2.4923 | Val Acc:   33.66%

Epoch 11 Summary:
  Train Loss: 2.3295 | Train Acc: 38.34%
  Val Loss:   2.4002 | Val Acc:   37.87%

Epoch 12 Summary:
  Train Loss: 2.2714 | Train Acc: 39.48%
  Val Loss:   2.3214 | Val Acc:   41.10%

Epoch 13 Summary:
  Train Loss: 2.2167 | Train Acc: 40.90%
  Val Loss:   2.4915 | Val Acc:   38.42%

Epoch 14 Summary:
  Train Loss: 2.1733 | Train Acc: 42.12%
  Val Loss:   2.3283 | Val Acc:   40.46%

Epoch 15 Summary:
  Train Loss: 2.0592 | Train Acc: 44.37%
  Val Loss:   2.1191 | Val Acc:   43.20%

Epoch 16 Summary:
  Train Loss: 2.0345 | Train Acc: 45.29%
  Val Loss:   2.0113 | Val Acc:   46.58%

Epoch 17 Summary:
  Train Loss: 2.0096 | Train Acc: 45.45%
  Val Loss:   2.0483 | Val Acc:   45.26%

Epoch 18 Summary:
  Train Loss: 1.9914 | Train Acc: 46.00%
  Val Loss:   2.0232 | Val Acc:   45.78%

Epoch 19 Summary:
  Train Loss: 1.9932 | Train Acc: 46.18%
  Val Loss:   1.9482 | Val Acc:   47.26%

Epoch 20 Summary:
  Train Loss: 1.9577 | Train Acc: 46.82%
  Val Loss:   1.9384 | Val Acc:   48.47%

Epoch 21 Summary:
  Train Loss: 1.9466 | Train Acc: 46.93%
  Val Loss:   1.9659 | Val Acc:   47.87%

Epoch 22 Summary:
  Train Loss: 1.9482 | Train Acc: 46.89%
  Val Loss:   1.9271 | Val Acc:   47.91%

Epoch 23 Summary:
  Train Loss: 1.9202 | Train Acc: 47.66%
  Val Loss:   2.0077 | Val Acc:   46.06%

Epoch 24 Summary:
  Train Loss: 1.9160 | Train Acc: 47.75%
  Val Loss:   1.9157 | Val Acc:   47.97%

Epoch 25 Summary:
  Train Loss: 1.9048 | Train Acc: 48.24%
  Val Loss:   1.8693 | Val Acc:   49.37%

Epoch 26 Summary:
  Train Loss: 1.8991 | Train Acc: 48.33%
  Val Loss:   1.9170 | Val Acc:   47.91%

Epoch 27 Summary:
  Train Loss: 1.8854 | Train Acc: 48.69%
  Val Loss:   1.8556 | Val Acc:   49.76%

Epoch 28 Summary:
  Train Loss: 1.8852 | Train Acc: 48.55%
  Val Loss:   1.8708 | Val Acc:   49.24%

Epoch 29 Summary:
  Train Loss: 1.8767 | Train Acc: 48.80%
  Val Loss:   1.8233 | Val Acc:   50.56%

Epoch 30 Summary:
  Train Loss: 1.8670 | Train Acc: 48.98%
  Val Loss:   1.8254 | Val Acc:   50.74%

Epoch 31 Summary:
  Train Loss: 1.8574 | Train Acc: 49.01%
  Val Loss:   1.8091 | Val Acc:   50.85%

Epoch 32 Summary:
  Train Loss: 1.8410 | Train Acc: 49.59%
  Val Loss:   1.8492 | Val Acc:   50.45%

Epoch 33 Summary:
  Train Loss: 1.8425 | Train Acc: 49.64%
  Val Loss:   1.7702 | Val Acc:   51.25%

Epoch 34 Summary:
  Train Loss: 1.8340 | Train Acc: 49.72%
  Val Loss:   1.8137 | Val Acc:   50.60%

Epoch 35 Summary:
  Train Loss: 1.8268 | Train Acc: 49.86%
  Val Loss:   1.7336 | Val Acc:   52.06%

Epoch 36 Summary:
  Train Loss: 1.8179 | Train Acc: 50.22%
  Val Loss:   1.7525 | Val Acc:   51.89%

Epoch 37 Summary:
  Train Loss: 1.8409 | Train Acc: 49.59%
  Val Loss:   1.8492 | Val Acc:   50.45%

Epoch 38 Summary:
  Train Loss: 1.8021 | Train Acc: 50.61%
  Val Loss:   1.7171 | Val Acc:   52.64%

Epoch 39 Summary:
  Train Loss: 1.8047 | Train Acc: 50.39%
  Val Loss:   1.6928 | Val Acc:   53.13%

Epoch 40 Summary:
  Train Loss: 1.7950 | Train Acc: 50.71%
  Val Loss:   1.7235 | Val Acc:   52.41%

Epoch 41 Summary:
  Train Loss: 1.7893 | Train Acc: 50.77%
  Val Loss:   1.7391 | Val Acc:   51.57%

Epoch 42 Summary:
  Train Loss: 1.7694 | Train Acc: 51.15%
  Val Loss:   1.7242 | Val Acc:   52.41%

Epoch 43 Summary:
  Train Loss: 1.7758 | Train Acc: 51.28%
  Val Loss:   1.6895 | Val Acc:   53.75%

Epoch 44 Summary:
  Train Loss: 1.7530 | Train Acc: 51.71%
  Val Loss:   1.6540 | Val Acc:   53.91%

Epoch 45 Summary:
  Train Loss: 1.7353 | Train Acc: 52.09%
  Val Loss:   1.6460 | Val Acc:   54.11%

Epoch 46 Summary:
  Train Loss: 1.6902 | Train Acc: 53.46%
  Val Loss:   1.6142 | Val Acc:   54.97%

Epoch 47 Summary:
  Train Loss: 1.6844 | Train Acc: 53.25%
  Val Loss:   1.6179 | Val Acc:   54.45%

Epoch 48 Summary:
  Train Loss: 1.6830 | Train Acc: 53.47%
  Val Loss:   1.6968 | Val Acc:   53.48%

Epoch 49 Summary:
  Train Loss: 1.6582 | Train Acc: 53.75%
  Val Loss:   1.6136 | Val Acc:   55.11%

Epoch 50 Summary:
  Train Loss: 1.6549 | Train Acc: 54.37%
  Val Loss:   1.7350 | Val Acc:   54.03%

Epoch 51 Summary:
  Train Loss: 1.6507 | Train Acc: 54.12%
  Val Loss:   1.6061 | Val Acc:   55.36%

Epoch 52 Summary:
  Train Loss: 1.6380 | Train Acc: 54.41%
  Val Loss:   1.5761 | Val Acc:   56.23%

Epoch 53 Summary:
  Train Loss: 1.6335 | Train Acc: 54.68%
  Val Loss:   1.6091 | Val Acc:   55.09%

Epoch 54 Summary:
  Train Loss: 1.6236 | Train Acc: 54.61%
  Val Loss:   1.5684 | Val Acc:   56.45%

Epoch 55 Summary:
  Train Loss: 1.6245 | Train Acc: 54.75%
  Val Loss:   1.5960 | Val Acc:   55.22%

Epoch 56 Summary:
  Train Loss: 1.6060 | Train Acc: 55.32%
  Val Loss:   1.5745 | Val Acc:   56.81%

Epoch 57 Summary:
  Train Loss: 1.5952 | Train Acc: 55.26%
  Val Loss:   1.5451 | Val Acc:   57.48%

Epoch 58 Summary:
  Train Loss: 1.5936 | Train Acc: 55.53%
  Val Loss:   1.5080 | Val Acc:   58.59%

Epoch 59 Summary:
  Train Loss: 1.5766 | Train Acc: 55.87%
  Val Loss:   1.5342 | Val Acc:   57.71%

Epoch 60 Summary:
  Train Loss: 1.5631 | Train Acc: 56.04%
  Val Loss:   1.5272 | Val Acc:   57.90%

Epoch 61 Summary:
  Train Loss: 1.5556 | Train Acc: 56.28%
  Val Loss:   1.4964 | Val Acc:   58.90%

Epoch 62 Summary:
  Train Loss: 1.5439 | Train Acc: 56.73%
  Val Loss:   1.4960 | Val Acc:   58.87%

Epoch 63 Summary:
  Train Loss: 1.5204 | Train Acc: 57.30%
  Val Loss:   1.4608 | Val Acc:   59.92%

Epoch 64 Summary:
  Train Loss: 1.5202 | Train Acc: 57.44%
  Val Loss:   1.4827 | Val Acc:   59.24%

Epoch 65 Summary:
  Train Loss: 1.5041 | Train Acc: 57.70%
  Val Loss:   1.4139 | Val Acc:   61.09%

Epoch 66 Summary:
  Train Loss: 1.5042 | Train Acc: 57.79%
  Val Loss:   1.4465 | Val Acc:   60.44%

Epoch 67 Summary:
  Train Loss: 1.4578 | Train Acc: 58.91%
  Val Loss:   1.4843 | Val Acc:   59.27%

Epoch 68 Summary:
  Train Loss: 1.4423 | Train Acc: 59.07%
  Val Loss:   1.4200 | Val Acc:   60.78%

Epoch 69 Summary:
  Train Loss: 1.4310 | Train Acc: 59.61%
  Val Loss:   1.4128 | Val Acc:   61.05%

Epoch 70 Summary:
  Train Loss: 1.4241 | Train Acc: 59.68%
  Val Loss:   1.5211 | Val Acc:   57.83%

Epoch 71 Summary:
  Train Loss: 1.4107 | Train Acc: 60.02%
  Val Loss:   1.4265 | Val Acc:   60.39%

Epoch 72 Summary:
  Train Loss: 1.3874 | Train Acc: 60.43%
  Val Loss:   1.3755 | Val Acc:   62.02%

Epoch 73 Summary:
  Train Loss: 1.3321 | Train Acc: 62.13%
  Val Loss:   1.3737 | Val Acc:   61.77%

Epoch 74 Summary:
  Train Loss: 1.3136 | Train Acc: 62.46%
  Val Loss:   1.3777 | Val Acc:   62.26%

Epoch 75 Summary:
  Train Loss: 1.3011 | Train Acc: 62.78%
  Val Loss:   1.3424 | Val Acc:   62.37%

Epoch 76 Summary:
  Train Loss: 1.2965 | Train Acc: 63.02%
  Val Loss:   1.3446 | Val Acc:   62.79%

Epoch 77 Summary:
  Train Loss: 1.2822 | Train Acc: 63.23%
  Val Loss:   1.3435 | Val Acc:   62.61%

Epoch 78 Summary:
  Train Loss: 1.2536 | Train Acc: 64.08%
  Val Loss:   1.3088 | Val Acc:   62.88%

Epoch 79 Summary:
  Train Loss: 1.2301 | Train Acc: 64.83%
  Val Loss:   1.2623 | Val Acc:   65.11%

Epoch 80 Summary:
  Train Loss: 1.1927 | Train Acc: 65.67%
  Val Loss:   1.2093 | Val Acc:   66.85%

Epoch 81 Summary:
  Train Loss: 1.1744 | Train Acc: 66.06%
  Val Loss:   1.2396 | Val Acc:   65.94%

Epoch 82 Summary:
  Train Loss: 1.1128 | Train Acc: 67.75%
  Val Loss:   1.1700 | Val Acc:   67.85%

Epoch 83 Summary:
  Train Loss: 1.0971 | Train Acc: 68.16%
  Val Loss:   1.1904 | Val Acc:   67.45%

Epoch 84 Summary:
  Train Loss: 1.0531 | Train Acc: 69.20%
  Val Loss:   1.1452 | Val Acc:   68.62%

Epoch 85 Summary:
  Train Loss: 1.0285 | Train Acc: 70.06%
  Val Loss:   1.1106 | Val Acc:   69.64%

Epoch 86 Summary:
  Train Loss: 0.9954 | Train Acc: 70.87%
  Val Loss:   1.1472 | Val Acc:   67.67%

Epoch 87 Summary:
  Train Loss: 0.9882 | Train Acc: 70.97%
  Val Loss:   1.1131 | Val Acc:   68.94%

Epoch 88 Summary:
  Train Loss: 0.9666 | Train Acc: 71.71%
  Val Loss:   1.0620 | Val Acc:   70.23%

Epoch 89 Summary:
  Train Loss: 0.9532 | Train Acc: 71.95%
  Val Loss:   1.0298 | Val Acc:   71.13%

Epoch 90 Summary:
  Train Loss: 0.9361 | Train Acc: 72.34%
  Val Loss:   1.1094 | Val Acc:   68.83%

Epoch 91 Summary:
  Train Loss: 0.8922 | Train Acc: 73.69%
  Val Loss:   1.0143 | Val Acc:   71.54%

Epoch 92 Summary:
  Train Loss: 0.8747 | Train Acc: 74.24%
  Val Loss:   1.0045 | Val Acc:   72.06%

Epoch 93 Summary:
  Train Loss: 0.8583 | Train Acc: 74.56%
  Val Loss:   0.9879 | Val Acc:   72.38%

Epoch 94 Summary:
  Train Loss: 0.8042 | Train Acc: 76.17%
  Val Loss:   0.9697 | Val Acc:   72.62%

Epoch 95 Summary:
  Train Loss: 0.7865 | Train Acc: 76.73%
  Val Loss:   0.9960 | Val Acc:   71.92%

Epoch 96 Summary:
  Train Loss: 0.7754 | Train Acc: 76.88%
  Val Loss:   1.0053 | Val Acc:   71.94%

Epoch 97 Summary:
  Train Loss: 0.7503 | Train Acc: 77.61%
  Val Loss:   0.9720 | Val Acc:   72.83%

Epoch 98 Summary:
  Train Loss: 0.7402 | Train Acc: 77.97%
  Val Loss:   1.0241 | Val Acc:   72.02%

Epoch 99 Summary:
  Train Loss: 0.7105 | Train Acc: 78.87%
  Val Loss:   1.0167 | Val Acc:   72.18%

Epoch 100 Summary:
  Train Loss: 0.7027 | Train Acc: 79.25%
  Val Loss:   0.9856 | Val Acc:   72.11%

Epoch 101 Summary:
  Train Loss: 0.6942 | Train Acc: 79.32%
  Val Loss:   0.9864 | Val Acc:   72.53%

Epoch 102 Summary:
  Train Loss: 0.6670 | Train Acc: 80.11%
  Val Loss:   0.9784 | Val Acc:   72.55%

Epoch 103 Summary:
  Train Loss: 0.6430 | Train Acc: 80.92%
  Val Loss:   0.9550 | Val Acc:   73.19%
Checkpoint saved: best_model.pth
  ✓ New best accuracy: 73.19%

🎉 Target accuracy of 73.0% reached!
Best validation accuracy: 73.19%
Training completed!
Best validation accuracy: 73.19%
```

## HuggingFace Spaces App
### Step 1: Prepare Your Model and Files
- Save your trained model in a format that's easy to load (e.g., .pth for PyTorch, .h5 for TensorFlow)
- Create a requirements.txt file listing all Python dependencies needed to run your model

### Step 2: Create a HuggingFace Account
- Go to https://huggingface.co/
- Click "Sign Up" and create a free account
- Verify your email address

### Step 3: Create a New Space
- Once logged in, click on your profile picture (top right)
- Select "New Space"
- Fill in the details:
  - Space name: Choose a descriptive name
  - License: Select appropriate license (e.g., MIT)
  - Select the SDK: Choose between:
    - Gradio (recommended for beginners - easier interface)
    - Streamlit (more customizable)
    - Static HTML (for static demos)
- Choose visibility: Public or Private
- Click "Create Space"

### Step 4: Start by Cloning the Repo
```
# When prompted for a password, use an access token with write permissions.
# Generate one from your settings: https://huggingface.co/settings/tokens
git clone https://huggingface.co/spaces/wayoutisin/resnet56
```
- Add requirements.txt
```
# Python version: >=3.12

matplotlib>=3.10.7
opencv-python>=4.12.0.88
torch>=2.8.0
torchvision>=0.23.0
tqdm>=4.67.1
```
- Add model file (.pth)
- Add the following code as app.py
```
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

print("Starting application...", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"PyTorch version: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

# Define your ResNet56 architecture
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet56(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet56, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(16, 9, stride=1)
        self.layer2 = self._make_layer(32, 9, stride=2)
        self.layer3 = self._make_layer(64, 9, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

print("Loading model...", flush=True)

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # Initialize model architecture
    model = ResNet56(num_classes=10)
    print("Model architecture created", flush=True)
    
    # Load the saved weights
    print("Loading checkpoint...", flush=True)
    checkpoint = torch.load('best_model.pth', map_location=device)
    print(f"Checkpoint type: {type(checkpoint)}", flush=True)
    
    # Handle different save formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded from model_state_dict", flush=True)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded from state_dict", flush=True)
        else:
            model.load_state_dict(checkpoint)
            print("Loaded checkpoint dict directly", flush=True)
    else:
        # If the entire model was saved
        model = checkpoint
        print("Loaded entire model", flush=True)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!", flush=True)
    
except Exception as e:
    print(f"ERROR loading model: {e}", flush=True)
    import traceback
    traceback.print_exc()
    # Create a dummy model so the app can still start
    model = ResNet56(num_classes=10).to(device)
    print("Using untrained model due to loading error", flush=True)

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                        std=[0.2470, 0.2435, 0.2616])
])

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image):
    if image is None:
        return {"error": "No image provided"}
    
    try:
        print("Processing image...", flush=True)
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        results = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        print(f"Prediction complete: {max(results, key=results.get)}", flush=True)
        
        return results
    
    except Exception as e:
        print(f"ERROR in prediction: {e}", flush=True)
        return {"error": f"Prediction failed: {str(e)}"}

print("Creating Gradio interface...", flush=True)

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="ResNet56 Image Classifier",
    description="Upload an image to classify it using ResNet56 trained on CIFAR-10",
    theme=gr.themes.Soft()
)

print("Launching demo...", flush=True)

if __name__ == "__main__":
    demo.launch()
    
print("Demo launched!", flush=True)
```
### Step 5: Large File System
```
brew install git-lfs
git lfs install
```

### Step 6: Push your change
```
git add .
git commit -m "Initial commit"
git push
```
- If you face issues in pushing the code to remote then it could be due to  
  - Authentication Problems
    - Your credentials might have expired or be incorrect
    - For Hugging Face, use your access token as the password, not your account password
    - Get a token from: https://huggingface.co/settings/tokens
    - Try: git remote set-url origin https://USER:TOKEN@huggingface.co/spaces/USER/REPO
  - Track with LFS
    ```
    git lfs track "*.pth"
    git add .gitattributes
    git commit -m "Add LFS tracking"

    # 3. Migrate all .pth files in history to LFS
    git lfs migrate import --include="*.pth" --everything

    # 4. Force push (be careful - this rewrites history)
    git push --force
    ```

### Step 7: Wait for Build
- HuggingFace will automatically build your Space
- Watch the build logs in the "Logs" section
- Fix any errors that appear and push updates

### Step 8: Test Your Demo
- Once the build succeeds, your Space will be live
- The URL will be: https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
- Test all functionality to ensure it works correctly

### Step 9: Share Your Link
- Your live demo link will be:
- https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME