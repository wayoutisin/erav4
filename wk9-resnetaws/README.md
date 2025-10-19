# ResNet18 Training on AWS EC2

A streamlined PyTorch ResNet18 training system with automated AWS EC2 deployment using AWS Toolkit for credential management.

## Features

- **Custom ResNet18 Implementation**: Train from scratch on CIFAR-10
- **AWS EC2 Automation**: One-command deployment to GPU instances (g4dn.xlarge)
- **AWS Toolkit Integration**: Seamless credential management using VS Code AWS Toolkit
- **Secure Configuration**: No secrets in code - uses boto3 default credential chain
- **Remote Training**: Automated file sync, remote execution, and monitoring

## Quick Start

### 1. Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### 2. AWS Setup

#### Step 1: Install AWS Toolkit

1. Open VS Code Extensions (Ctrl+Shift+X)
2. Search for "AWS Toolkit"
3. Install the extension

#### Step 2: Connect to AWS

1. Press `Ctrl+Shift+P`
2. Type "AWS: Connect to AWS"
3. Choose one of:
   - **Builder ID** (free, no AWS account needed for basic features)
   - **IAM Identity Center** (enterprise SSO)
   - **IAM Credentials** (access key + secret)

The AWS Toolkit will manage your credentials automatically. No manual configuration needed!

#### Step 3: Configure EC2 SSH Keys

Edit `configs/aws_config.json`:

```json
{
  "ec2": {
    "key_name": "your-ec2-key-name",
    "key_path": "C:/Users/YourName/.ssh/your-key.pem"
  }
}
```

**Important**: 
- Create your EC2 key pair in AWS Console: EC2 → Key Pairs → Create
- Download the `.pem` file and save it securely
- On Windows: Use forward slashes or double backslashes in paths
- Set correct permissions: `chmod 400 ~/.ssh/your-key.pem`

### 3. Verify Setup

```bash
python src/utils/config_utils.py
```

This validates:
- ✓ AWS credentials accessible via AWS Toolkit
- ✓ EC2 SSH key configured correctly
- ✓ Configuration files loaded

## Usage

### Local Training

Train ResNet18 locally on CIFAR-10:

```bash
python src/train.py --epochs 10 --batch-size 128 --lr 0.1
```

**Options:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.1)
- `--data-dir`: Dataset directory (default: ./data)
- `--checkpoint-dir`: Checkpoint save directory (default: ./checkpoints)
- `--resume`: Resume from checkpoint path

### AWS EC2 Training

Deploy and train on AWS EC2 GPU instance:

```bash
python -m src.deployment.orchestrator \
  --aws-config configs/aws_config.json \
  --training-config configs/training_config.json \
  --instance_id <<INSTANCE_ID>>\
  --public_ip <<INSTANCE_PUBLIC_IP_ADDRESS>>
```

**Workflow:**
1. Creates EC2 g4dn.xlarge instance with GPU (NVIDIA T4)
2. Syncs code and data to instance
3. Installs dependencies automatically
4. Runs training remotely
5. Syncs checkpoints back to local machine

**Common Commands:**

```bash
# Launch instance only
python src/deployment/orchestrator.py --action launch

# Check instance status
python src/deployment/orchestrator.py --action status

# Sync files to instance
python src/deployment/orchestrator.py --action sync

# Run training on existing instance
python src/deployment/orchestrator.py --action remote-train --epochs 50

# Terminate instance
python src/deployment/orchestrator.py --action terminate
```

**Cost Management:**
- g4dn.xlarge: ~$0.526/hour (us-east-1, on-demand)
- Always terminate instances when done: `--action terminate`
- Use `--action status` to check running instances

## Project Structure

```
resnet/
├── src/
│   ├── models/
│   │   └── resnet.py              # Custom ResNet18 implementation
│   ├── training/
│   │   ├── trainer.py             # Training loop with mixed precision
│   │   └── evaluator.py           # Model evaluation
│   ├── data/
│   │   └── dataset.py             # CIFAR-10 data loading
│   ├── deployment/
│   │   ├── ec2_manager.py         # EC2 instance management
│   │   └── orchestrator.py        # End-to-end deployment workflow
│   ├── utils/
│   │   ├── config_utils.py        # Configuration & AWS Toolkit integration
│   │   └── logger_utils.py        # Logging utilities
│   └── train.py                   # Main training script
├── configs/
│   ├── aws_config.json            # AWS EC2 configuration
│   └── training_config.json       # Training hyperparameters
├── requirements.txt
└── README.md
```

## Configuration Files

### `configs/aws_config.json`

```json
{
  "aws": {
    "region": "us-east-1"
  },
  "ec2": {
    "ami_id": "ami-0c02fb55d7a07f4c6",
    "instance_type": "g4dn.xlarge",
    "key_name": "your-key-name",
    "key_path": "/path/to/your-key.pem",
    "security_groups": ["default"],
    "storage": {
      "volume_size": 100,
      "volume_type": "gp3"
    }
  }
}
```

**Required Changes:**
- `ec2.key_name`: Your EC2 key pair name
- `ec2.key_path`: Path to your .pem file

**Optional Changes:**
- `aws.region`: AWS region (default: us-east-1)
- `ec2.instance_type`: Instance type (default: g4dn.xlarge)
- `ec2.ami_id`: Deep Learning AMI (default: Ubuntu 20.04)

### `configs/training_config.json`

```json
{
  "training": {
    "epochs": 50,
    "optimizer": {
      "type": "sgd",
      "lr": 0.1,
      "momentum": 0.9,
      "weight_decay": 0.0001
    },
    "scheduler": {
      "type": "cosine",
      "eta_min": 0.0001
    }
  },
  "data": {
    "dataset": "cifar10",
    "batch_size": 128,
    "num_workers": 4
  }
}
```

## AWS Toolkit Credential Management

### How It Works

The codebase uses **boto3's default credential chain**, which automatically discovers credentials from:

1. **AWS Toolkit** (recommended) - automatic when connected in VS Code
2. **AWS CLI** - credentials from `~/.aws/credentials`
3. **Environment variables** - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
4. **IAM roles** - automatic when running on EC2

**No manual credential configuration needed!** Just connect AWS Toolkit and you're ready.

### Switching AWS Profiles

If you have multiple AWS accounts:

```bash
# Set profile via environment variable
set AWS_PROFILE=my-profile  # Windows
export AWS_PROFILE=my-profile  # Linux/Mac

# Or configure in configs/aws_config.json
{
  "aws": {
    "profile": "my-profile"
  }
}
```

### Credential Verification

```bash
# Verify credentials are working
python -c "import boto3; print(boto3.client('sts').get_caller_identity())"

# Or use the configuration validator
python src/utils/config_utils.py
```

### Troubleshooting Credentials

**Issue**: `NoCredentialsError` or `Unable to locate credentials`

**Solutions**:

1. **Check AWS Toolkit Connection**:
   - Open AWS Toolkit panel in VS Code (AWS icon in left sidebar)
   - Ensure you see "Connected to AWS" with your account
   - If not connected, click "Connect to AWS" and authenticate

2. **Verify AWS CLI** (alternative):
   ```bash
   aws configure
   # Enter: Access Key ID, Secret Access Key, Region
   ```

3. **Check Credential File**:
   - Location: `C:\Users\YourName\.aws\credentials` (Windows)
   - Should contain:
     ```
     [default]
     aws_access_key_id = YOUR_KEY
     aws_secret_access_key = YOUR_SECRET
     ```

4. **Test Credentials**:
   ```bash
   aws sts get-caller-identity
   ```

## EC2 Training Workflow Details

### Initial Setup

```bash
# 1. Launch instance
python src/deployment/orchestrator.py --action launch

# Wait for instance to be running (~2-3 minutes)
```

### File Synchronization

```bash
# 2. Sync code and data to EC2
python src/deployment/orchestrator.py --action sync

# This syncs:
# - Source code (src/)
# - Configuration (configs/)
# - Requirements (requirements.txt)
# - Dataset (downloads CIFAR-10 automatically if needed)
```

### Remote Training

```bash
# 3. Train on EC2
python src/deployment/orchestrator.py --action remote-train --epochs 50

# Monitors training in real-time
# Checkpoints saved to /home/ubuntu/resnet/checkpoints/ on EC2
```

### Checkpoint Retrieval

```bash
# 4. Sync results back to local
python src/deployment/orchestrator.py --action sync-results

# Downloads:
# - Model checkpoints
# - Training logs
# - Evaluation metrics
```

### Cleanup

```bash
# 5. Terminate instance
python src/deployment/orchestrator.py --action terminate

# IMPORTANT: Always terminate to avoid charges!
```

### All-in-One Command

```bash
# Launch, sync, train, retrieve, terminate
python src/deployment/orchestrator.py --action launch --remote-train --epochs 50 --auto-terminate
```

## Model Architecture

Custom ResNet18 implementation with:

- **Residual blocks**: 2x for each stage [64, 128, 256, 512]
- **Adaptive pooling**: Global average pooling before classifier
- **Batch normalization**: After each convolutional layer
- **ReLU activation**: Standard non-linearity
- **Shortcut connections**: Identity and projection shortcuts

**Architecture Summary**:
```
Input (3x32x32) 
→ Conv1 (64) 
→ [ResBlock(64)]×2 
→ [ResBlock(128)]×2 
→ [ResBlock(256)]×2 
→ [ResBlock(512)]×2 
→ AdaptiveAvgPool 
→ FC(10)
Output (10 classes)
```

**Parameters**: ~11M trainable parameters

## Training Performance

**Local (CPU)**: 
- ~300-400 seconds/epoch
- Recommended for testing only

**AWS EC2 (g4dn.xlarge with T4 GPU)**:
- ~15-20 seconds/epoch
- ~15-20x faster than CPU
- Expected accuracy: ~92-94% on CIFAR-10

**Memory Usage**:
- Training: ~2-3GB GPU memory
- Batch size 128: Safe for 16GB T4 GPU

## Advanced Usage

### Custom Dataset

```python
# Modify src/data/dataset.py
def get_dataloaders(data_dir, batch_size, dataset='cifar10'):
    if dataset == 'my_dataset':
        # Add your dataset loading logic
        pass
```

### Hyperparameter Tuning

Edit `configs/training_config.json` or use command-line:

```bash
python src/train.py \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.05 \
  --weight-decay 0.0005
```

### Resume Training

```bash
# Resume from checkpoint
python src/train.py --resume checkpoints/checkpoint_epoch_20.pth
```

### Multi-GPU Training (EC2 p3 instances)

```python
# Modify src/train.py for DataParallel
model = nn.DataParallel(model)
```

## Monitoring & Logging

### Local Logs

- Training logs: Console output
- Checkpoints: `./checkpoints/`
- Best model: `./checkpoints/best_model.pth`

### EC2 Logs

- Remote logs: `/home/ubuntu/resnet/logs/`
- View real-time: `--action remote-train` shows live output
- SSH access: `ssh -i your-key.pem ubuntu@<instance-ip>`

### CloudWatch (Optional)

Enable in `configs/aws_config.json`:
```json
{
  "monitoring": {
    "cloudwatch_enabled": true,
    "log_group": "/aws/ec2/ml-training"
  }
}
```

## Troubleshooting

### AWS Toolkit Issues

**Problem**: AWS Toolkit not showing credentials

**Solution**:
1. Restart VS Code
2. Press Ctrl+Shift+P → "AWS: Logout" then reconnect
3. Check AWS Toolkit output panel for errors

### EC2 Connection Issues

**Problem**: Cannot SSH to EC2 instance

**Solutions**:
1. Check security group allows SSH (port 22) from your IP
2. Verify key permissions: `icacls your-key.pem /inheritance:r /grant:r "%USERNAME%:R"`
3. Wait 2-3 minutes after launch for instance initialization
4. Check instance status: `--action status`

**Problem**: "Permission denied (publickey)" error

**Solutions**:
1. Verify key_path in `configs/aws_config.json` is correct
2. Ensure key_name matches EC2 console
3. Check AMI uses correct username (Ubuntu AMI uses `ubuntu`)

### Training Issues

**Problem**: CUDA out of memory

**Solutions**:
1. Reduce batch size: `--batch-size 64`
2. Use smaller instance: Check current usage with `nvidia-smi` on EC2
3. Clear GPU cache periodically in training loop

**Problem**: Low training accuracy

**Solutions**:
1. Verify data augmentation is enabled (check `src/data/dataset.py`)
2. Try lower learning rate: `--lr 0.01`
3. Increase epochs: `--epochs 100`
4. Check model architecture matches CIFAR-10 (10 classes)

### Cost Management

**Problem**: Forgot to terminate instance

**Solution**:
```bash
# List running instances
python src/deployment/orchestrator.py --action list

# Terminate specific instance
python src/deployment/orchestrator.py --action terminate --instance-id i-xxxxx
```

**Prevention**:
- Use `--auto-terminate` flag
- Set up AWS billing alerts in AWS Console
- Use spot instances (add to `configs/aws_config.json`)

## Security Best Practices

### ✓ What This Project Does

1. **No secrets in code**: Uses AWS Toolkit/boto3 credential chain
2. **SSH key protection**: Keys stored outside repository
3. **Security groups**: Restrict access to your IP only
4. **IAM roles**: Instance profile for secure AWS API access
5. **Encrypted storage**: EBS volumes encrypted by default

### ⚠️ What You Should Do

1. **Never commit**:
   - `.pem` files
   - AWS credentials
   - `configs/aws_config.json` with your key paths (add to `.gitignore`)

2. **Restrict access**:
   - Update security group to allow SSH only from your IP
   - Regularly rotate IAM credentials
   - Use IAM roles with minimal permissions

3. **Monitor usage**:
   - Set up AWS billing alerts
   - Review CloudTrail logs periodically
   - Terminate unused instances

## Cost Estimation

### Training 50 Epochs on CIFAR-10

**Instance**: g4dn.xlarge ($0.526/hour in us-east-1)

- Training time: ~50 epochs × 20 sec/epoch = ~17 minutes
- Setup/teardown: ~5 minutes
- **Total time**: ~22 minutes = ~0.37 hours
- **Cost**: ~$0.20

**Storage**: 100GB EBS gp3 ($0.08/GB-month)
- Per day: ~$0.26
- Per hour: ~$0.011

**Total estimated cost**: ~$0.21 per training run

### Monthly Development Estimate

- 10 training runs: ~$2.10
- Storage (continuous): ~$8.00
- Data transfer: ~$0.50
- **Total**: ~$10.60/month (development usage)

**Tips to reduce costs**:
- Use Spot Instances (up to 70% cheaper)
- Terminate instances immediately after use
- Use smaller storage volumes
- Use S3 for dataset storage (cheaper than EBS)

## Requirements

### Python Packages

```
torch>=2.0.0
torchvision>=0.15.0
boto3>=1.26.0
paramiko>=3.0.0
numpy>=1.24.0
```

### AWS Services

- **EC2**: GPU instance (g4dn.xlarge recommended)
- **STS**: Credential verification
- **SSM**: Parameter store (optional, for secrets)
- **CloudWatch**: Logging and monitoring (optional)

### AWS IAM Permissions Required

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:CreateTags",
        "ec2:DescribeImages",
        "ec2:DescribeKeyPairs",
        "ec2:DescribeSecurityGroups"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
```

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review AWS Toolkit documentation
3. Check boto3 documentation for credential setup
4. Open an issue with detailed error messages

---

**Happy Training!** 🚀

Remember to:
- ✓ Connect AWS Toolkit before starting
- ✓ Configure EC2 SSH keys
- ✓ Always terminate instances when done
- ✓ Monitor AWS costs regularly
