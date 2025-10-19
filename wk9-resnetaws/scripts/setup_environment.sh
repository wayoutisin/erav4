#!/bin/bash
# Setup script for AWS EC2 instance environment

set -e

echo "Setting up ResNet18 training environment on EC2..."

# Update system
sudo apt-get update -y
sudo apt-get upgrade -y

# Install essential packages
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    htop \
    nvtop \
    tmux \
    vim \
    curl \
    wget \
    unzip

# Install AWS CLI v2
echo "Installing AWS CLI v2..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Install Docker (optional but useful)
echo "Installing Docker..."
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Install Miniconda
echo "Installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh

# Initialize conda
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
$HOME/miniconda3/bin/conda init bash

# Create conda environment for PyTorch
echo "Creating PyTorch environment..."
$HOME/miniconda3/bin/conda create -n pytorch-env python=3.8 -y

# Activate environment and install PyTorch with CUDA support
echo "Installing PyTorch and dependencies..."
source $HOME/miniconda3/bin/activate pytorch-env

# Install PyTorch with CUDA support (adjust version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install additional Python packages
pip install --upgrade pip
pip install \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    boto3 \
    psutil \
    jupyter \
    ipywidgets

# Install optional packages
pip install \
    tensorboard \
    onnx \
    onnxruntime-gpu

echo "Environment setup completed!"

# Create workspace directory
mkdir -p ~/resnet-training
cd ~/resnet-training

# Create necessary directories
mkdir -p data models outputs logs scripts

# Set up git (optional)
echo "Configuring git..."
read -p "Enter your git username (optional): " git_username
read -p "Enter your git email (optional): " git_email

if [ ! -z "$git_username" ]; then
    git config --global user.name "$git_username"
fi

if [ ! -z "$git_email" ]; then
    git config --global user.email "$git_email"
fi

# Create a simple system monitoring script
cat > ~/monitor_training.sh << 'EOF'
#!/bin/bash
echo "=== System Monitoring ==="
echo "Date: $(date)"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
echo ""
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
echo ""
echo "Memory Usage:"
free -h | grep "Mem:" | awk '{print "Used: " $3 "/" $2 " (" $3/$2*100 "%)"}'
echo ""
echo "Disk Usage:"
df -h / | tail -1 | awk '{print "Used: " $3 "/" $2 " (" $5 ")"}'
echo ""
echo "Running Processes:"
ps aux | grep python | grep -v grep
echo "========================="
EOF

chmod +x ~/monitor_training.sh

# Create training status checker
cat > ~/check_training.sh << 'EOF'
#!/bin/bash
LOG_FILE="$HOME/resnet-training/logs/training.log"

if [ -f "$LOG_FILE" ]; then
    echo "=== Latest Training Progress ==="
    tail -20 "$LOG_FILE" | grep -E "(Epoch|Train Loss|Val Loss|Best|Error|Complete)"
    echo ""
    echo "=== Training Status ==="
    if pgrep -f "train.py" > /dev/null; then
        echo "Status: Training is running"
        echo "PID: $(pgrep -f train.py)"
    else
        echo "Status: No training process found"
    fi
else
    echo "No training log found at $LOG_FILE"
fi
EOF

chmod +x ~/check_training.sh

# Create cleanup script
cat > ~/cleanup_training.sh << 'EOF'
#!/bin/bash
echo "Cleaning up training artifacts..."

# Remove large output files (optional)
read -p "Remove output files? (y/n): " confirm
if [ "$confirm" = "y" ]; then
    rm -rf ~/resnet-training/outputs/*
    echo "Output files removed"
fi

# Remove model checkpoints except best model
read -p "Remove model checkpoints (keeping best model)? (y/n): " confirm
if [ "$confirm" = "y" ]; then
    find ~/resnet-training/models -name "*.pth" ! -name "*best*" -delete
    echo "Model checkpoints cleaned"
fi

# Remove cached data
read -p "Remove cached data? (y/n): " confirm
if [ "$confirm" = "y" ]; then
    rm -rf ~/.cache/torch
    rm -rf ~/resnet-training/data/__pycache__
    echo "Cache cleaned"
fi

echo "Cleanup completed!"
EOF

chmod +x ~/cleanup_training.sh

echo ""
echo "=== Setup Summary ==="
echo "✓ System packages installed"
echo "✓ AWS CLI v2 installed"
echo "✓ Docker installed"
echo "✓ Miniconda installed"
echo "✓ PyTorch environment created"
echo "✓ Workspace directory created: ~/resnet-training"
echo "✓ Monitoring scripts created"
echo ""
echo "=== Next Steps ==="
echo "1. Source your bashrc: source ~/.bashrc"
echo "2. Activate environment: conda activate pytorch-env"
echo "3. Clone your project to ~/resnet-training"
echo "4. Start training!"
echo ""
echo "=== Useful Commands ==="
echo "Monitor system: ~/monitor_training.sh"
echo "Check training: ~/check_training.sh"
echo "Cleanup: ~/cleanup_training.sh"
echo ""
echo "Setup completed successfully! 🎉"