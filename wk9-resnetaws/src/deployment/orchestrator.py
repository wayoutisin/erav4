#!/usr/bin/env python3
"""
Deployment orchestrator for training ResNet18 on existing AWS EC2 GPU instance
Optimized for Deep Learning OSS Nvidia Driver AMI with PyTorch pre-installed
"""
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional

from .file_sync import FileSync
from .remote_runner import RemoteRunner
from ..utils.logger_utils import setup_logger
from ..utils.config_utils import SecureConfigLoader


class DeploymentOrchestrator:
    def __init__(self, aws_config_path: str = None, training_config_path: str = None, 
                 instance_id: str = None, public_ip: str = None):
        """Initialize deployment orchestrator for existing GPU instance"""
        
        config_loader = SecureConfigLoader()
        
        try:
            self.aws_config = config_loader.load_aws_config()
            self.training_config = config_loader.load_training_config()
            self.logger = setup_logger('deployment')
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger = setup_logger('deployment')
            self.logger.error(f"Failed to load configuration: {e}")
            config_loader.print_required_env_vars()
            raise
        
        self.aws_config_path = aws_config_path
        self.training_config_path = training_config_path
        
        self.instance_id = instance_id
        self.instance_info = {
            'instance_id': instance_id,
            'public_ip': public_ip
        }
        
        self.file_sync = FileSync(aws_config_path)
        self.remote_runner = None
        
        self._validate_instance_info()
    
    def _validate_instance_info(self):
        """Validate instance information"""
        if not self.instance_id or not self.instance_info.get('public_ip'):
            raise ValueError(
                "Instance ID and public IP are required.\n"
                "Provide via --instance-id and --public-ip arguments."
            )
        
        self.logger.info(f"Target instance: {self.instance_id}")
        self.logger.info(f"Public IP: {self.instance_info['public_ip']}")
    
    def deploy_and_train(self, 
                        sync_code: bool = True,
                        sync_data: bool = True, 
                        install_requirements: bool = True,
                        start_training: bool = True,
                        monitor: bool = True) -> dict:
        """Training pipeline on GPU instance with PyTorch pre-installed"""
        
        results = {
            'success': False,
            'instance_id': self.instance_id,
            'instance_info': self.instance_info,
            'training_results': None,
            'error': None
        }
        
        try:
            # Connect to instance
            self.logger.info("Connecting to remote instance...")
            self.remote_runner = RemoteRunner(
                self.instance_info['public_ip'],
                self.aws_config['ec2']['key_path'],
                self.aws_config
            )
            
            if not self._verify_connection():
                raise Exception("Failed to connect to instance via SSH")
            
            # Verify GPU and PyTorch
            self._verify_gpu_and_pytorch()
            
            # Step 1: Sync code first
            if sync_code:
                self.logger.info("Syncing code to remote instance...")
                if not self.file_sync.sync_to_remote(
                    self.instance_info['public_ip'], 
                    self.aws_config['ec2']['key_name']
                ):
                    raise Exception("Failed to sync code")
            
            # Create workspace directories
            self.logger.info("Creating workspace directories...")
            remote_workspace = self.aws_config['deployment']['remote_workspace']
            mkdir_cmd = f"mkdir -p {remote_workspace}/logs {remote_workspace}/outputs {remote_workspace}/data"
            self.remote_runner.execute_command(mkdir_cmd, wait_for_completion=True)
            
            # Step 2: Install additional requirements if needed
            if install_requirements:
                self.logger.info("Installing additional requirements...")
                if not self._install_requirements():
                    self.logger.warning("Failed to install requirements, continuing anyway...")
            
            # Step 3: Sync data
            if sync_data:
                self.logger.info("Syncing training data...")
                if not self._sync_training_data():
                    raise Exception("Failed to sync training data")
            
            # Step 4: Start training
            if start_training:
                self.logger.info("Starting training...")
                training_results = self._start_remote_training()
                results['training_results'] = training_results
                
                if not training_results['success']:
                    raise Exception(f"Training failed: {training_results.get('error', 'Unknown error')}")
            
            # Step 5: Monitor training
            if monitor and start_training:
                self.logger.info("Monitoring training progress...")
                self._monitor_training()
            
            results['success'] = True
            self.logger.info("Deployment and training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _verify_connection(self) -> bool:
        """Verify SSH connection"""
        try:
            result = self.remote_runner.execute_command("echo 'Connected'", wait_for_completion=True)
            self.logger.info("SSH connection verified")
            return result is not None
        except Exception as e:
            self.logger.error(f"SSH connection failed: {e}")
            return False
    
    def _verify_gpu_and_pytorch(self):
        """Verify GPU and PyTorch are available"""
        self.logger.info("Verifying GPU and PyTorch installation...")
        
        # Check GPU
        gpu_check = self.remote_runner.execute_command("nvidia-smi", wait_for_completion=True)
        if gpu_check:
            self.logger.info("✓ GPU detected")
        else:
            self.logger.warning("GPU check failed")
        
        # Check PyTorch
        pytorch_check = self.remote_runner.execute_command(
            "python3 -c \"import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')\"",
            wait_for_completion=True
        )
        if pytorch_check:
            self.logger.info(f"✓ PyTorch available: {pytorch_check.strip()}")
        else:
            self.logger.warning("PyTorch check failed")
    
    def _install_requirements(self) -> bool:
        """Install additional Python requirements"""
        requirements_file = self.aws_config['deployment'].get('requirements_file')
        if not requirements_file:
            self.logger.info("No requirements file specified, skipping")
            return True
        
        remote_workspace = self.aws_config['deployment']['remote_workspace']
        remote_req_path = f"{remote_workspace}/{requirements_file}"
        
        # Check if requirements file exists
        check_cmd = f"test -f {remote_req_path} && echo 'exists' || echo 'not found'"
        check_result = self.remote_runner.execute_command(check_cmd, wait_for_completion=True)
        
        if not check_result or 'not found' in check_result:
            self.logger.warning(f"Requirements file not found at {remote_req_path}")
            return True  # Not a fatal error
        
        # Create virtual environment first
        self.logger.info("Creating Python virtual environment...")
        venv_cmd = f"""
        python3 -m venv {remote_workspace}/venv && \
        source {remote_workspace}/venv/bin/activate && \
        pip install --upgrade pip
        """
        success = self.remote_runner.execute_command(venv_cmd, wait_for_completion=True, timeout=120)
        if not success:
            self.logger.error("Failed to create virtual environment")
            return False
        
        # Install requirements in virtual environment
        self.logger.info("Installing requirements...")
        cmd = f"""
        source {remote_workspace}/venv/bin/activate && \
        cd {remote_workspace} && \
        pip install -r {requirements_file}
        """
        
        success = self.remote_runner.execute_command(cmd, wait_for_completion=True, timeout=600)
        if not success:
            self.logger.error("Failed to install requirements")
            return False
        
        self.logger.info("Requirements installed successfully in virtual environment")
        return True
    
    def _sync_training_data(self) -> bool:
        """Sync training data to remote instance"""
        local_data_path = Path(self.training_config.get('data', {}).get('dataset_path', ''))
        
        if not local_data_path.exists():
            self.logger.warning(f"Local data path does not exist: {local_data_path}")
            return True  # Not an error
        
        remote_data_path = f"{self.aws_config['deployment']['remote_workspace']}/data"
        
        return self.file_sync.sync_directory(
            str(local_data_path),
            f"{self.instance_info['public_ip']}:{remote_data_path}",
            self.aws_config['ec2']['key_path']
        )
    
    def _start_remote_training(self) -> dict:
        """Start training on remote instance"""
        remote_workspace = self.aws_config['deployment']['remote_workspace']
        
        # Upload training config
        remote_config_path = f"{remote_workspace}/training_config.json"
        self.file_sync.upload_file(
            self.training_config_path,
            self.instance_info['public_ip'],
            remote_config_path,
            self.aws_config['ec2']['key_name']
        )
        
        # Start training command using virtual environment
        training_cmd = f"""
        source {remote_workspace}/venv/bin/activate && \
        cd {remote_workspace} && \
        python -m src.training.train --config training_config.json --output-dir outputs 2>&1 | tee logs/training.log
        """
        
        job_id = self.remote_runner.execute_command(
            training_cmd, 
            background=True,
            wait_for_completion=False
        )
        
        if job_id:
            return {
                'success': True,
                'job_id': job_id,
                'log_file': f"{remote_workspace}/logs/training.log"
            }
        else:
            return {
                'success': False,
                'error': 'Failed to start training job'
            }
    
    def _monitor_training(self):
        """Monitor training progress"""
        remote_workspace = self.aws_config['deployment']['remote_workspace']
        log_file = f"{remote_workspace}/logs/training.log"
        
        self.logger.info("Training started. Monitor with:")
        self.logger.info(f"ssh -i {self.aws_config['ec2']['key_path']} ubuntu@{self.instance_info['public_ip']} 'tail -f {log_file}'")
        
        if input("Stream training logs? (y/n): ").lower() == 'y':
            self._stream_training_logs(log_file)
    
    def _stream_training_logs(self, log_file: str):
        """Stream training logs in real-time"""
        ssh_cmd = [
            'ssh', 
            '-i', self.aws_config['ec2']['key_path'],
            '-o', 'StrictHostKeyChecking=no',
            f"ubuntu@{self.instance_info['public_ip']}",
            f'tail -f {log_file}'
        ]
        
        try:
            process = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True)
            
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())
                
        except KeyboardInterrupt:
            self.logger.info("Log streaming stopped")
            process.terminate()
        except Exception as e:
            self.logger.error(f"Error streaming logs: {e}")
    
    def download_results(self, local_output_dir: str = "results/"):
        """Download training results"""
        local_output_path = Path(local_output_dir)
        local_output_path.mkdir(parents=True, exist_ok=True)
        
        remote_workspace = self.aws_config['deployment']['remote_workspace']
        
        outputs_downloaded = self.file_sync.sync_directory(
            f"{self.instance_info['public_ip']}:{remote_workspace}/outputs/",
            str(local_output_path / "outputs"),
            self.aws_config['ec2']['key_name'],
            direction='from_remote'
        )
        
        logs_downloaded = self.file_sync.sync_directory(
            f"{self.instance_info['public_ip']}:{remote_workspace}/logs/",
            str(local_output_path / "logs"),
            self.aws_config['ec2']['key_name'],
            direction='from_remote'
        )
        
        if outputs_downloaded and logs_downloaded:
            self.logger.info(f"Results downloaded to: {local_output_path}")
            return True
        else:
            self.logger.error("Failed to download some results")
            return False


def main():
    parser = argparse.ArgumentParser(description='Train on AWS GPU instance (PyTorch pre-installed)')
    parser.add_argument('--aws-config', type=str, required=True, 
                       help='AWS configuration file')
    parser.add_argument('--training-config', type=str, required=True,
                       help='Training configuration file')
    parser.add_argument('--instance-id', type=str, required=True,
                       help='EC2 instance ID')
    parser.add_argument('--public-ip', type=str, required=True,
                       help='Public IP of EC2 instance')
    parser.add_argument('--no-sync-code', action='store_true',
                       help='Skip syncing code')
    parser.add_argument('--no-sync-data', action='store_true',
                       help='Skip syncing data')
    parser.add_argument('--no-requirements', action='store_true',
                       help='Skip installing requirements')
    parser.add_argument('--no-training', action='store_true',
                       help='Skip training')
    parser.add_argument('--no-monitor', action='store_true',
                       help='Skip monitoring')
    parser.add_argument('--download-results', type=str,
                       help='Download results to directory')
    
    args = parser.parse_args()
    
    orchestrator = DeploymentOrchestrator(
        args.aws_config, 
        args.training_config,
        instance_id=args.instance_id,
        public_ip=args.public_ip
    )
    
    try:
        results = orchestrator.deploy_and_train(
            sync_code=not args.no_sync_code,
            sync_data=not args.no_sync_data,
            install_requirements=not args.no_requirements,
            start_training=not args.no_training,
            monitor=not args.no_monitor
        )
        
        print("\nDeployment Results:")
        print(json.dumps(results, indent=2, default=str))
        
        if args.download_results and results['success']:
            orchestrator.download_results(args.download_results)
    
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
    except Exception as e:
        print(f"Deployment failed: {e}")


if __name__ == '__main__':
    main()