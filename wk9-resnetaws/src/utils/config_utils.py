"""
Configuration management for AWS deployment using AWS Toolkit.

This module loads configuration from JSON files and uses boto3's default
credential chain for AWS authentication (AWS Toolkit, AWS CLI, IAM roles).

Only EC2 SSH keys need explicit configuration.
"""

import os
import json
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class SecureConfigLoader:
    """
    Loads configuration using boto3's default credential chain.
    
    AWS credentials are automatically discovered from:
    - AWS Toolkit for VS Code
    - AWS CLI profiles (~/.aws/config and ~/.aws/credentials)
    - Environment variables
    - IAM roles (when running on EC2)
    
    Only EC2 SSH keys need explicit configuration.
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._log_credential_source()
    
    def _log_credential_source(self):
        """Log which credential source is being used."""
        if os.getenv('AWS_PROFILE'):
            logger.info(f"AWS credentials: Using profile '{os.getenv('AWS_PROFILE')}'")
        else:
            logger.info("AWS credentials: Using boto3 default credential chain (AWS Toolkit/CLI)")
    
    def load_aws_config(self) -> Dict[str, Any]:
        """
        Load AWS configuration from JSON file.
        
        AWS credentials are automatically discovered by boto3.
        Only EC2 SSH key information needs explicit configuration.
        """
        # Load base configuration from JSON
        aws_config_file = self.config_dir / "aws_config.json"
        with open(aws_config_file, 'r') as f:
            config = json.load(f)
        
        # Override with environment variables if set (optional)
        # env_mappings = {
        #     'aws.region': 'AWS_DEFAULT_REGION',
        #     'aws.profile': 'AWS_PROFILE',
        #     'ec2.key_name': 'AWS_EC2_KEY_NAME',
        #     'ec2.key_path': 'AWS_EC2_KEY_PATH',
        # }
        
        # for config_path, env_var in env_mappings.items():
        #     env_value = os.getenv(env_var)
        #     if env_value:
        #         self._set_nested_config(config, config_path, env_value)
        
        # Validate required EC2 SSH configuration
        self._validate_ec2_config(config)
        
        logger.info("AWS configuration loaded successfully")
        return config
    
    def _validate_ec2_config(self, config: Dict[str, Any]):
        """Validate required EC2 SSH configuration."""
        ec2_config = config.get('ec2', {})
        # Check for EC2 key name
        if not ec2_config.get('key_name'):
            raise ValueError(
                "EC2 key name not configured. Set in configs/aws_config.json:\n"
                '  {"ec2": {"key_name": "your-key-name"}}\n'
                "Or set environment variable: AWS_EC2_KEY_NAME"
            )
        
        # Check for EC2 key path
        key_path = ec2_config.get('key_path')
        if not key_path:
            raise ValueError(
                "EC2 key path not configured. Set in configs/aws_config.json:\n"
                '  {"ec2": {"key_path": "/path/to/key.pem"}}\n'
                "Or set environment variable: AWS_EC2_KEY_PATH"
            )
        
        # Validate key file exists
        key_path_obj = Path(key_path)
        if not key_path_obj.exists():
            raise FileNotFoundError(
                f"EC2 key file not found: {key_path}\n"
                f"Please ensure the .pem file exists at this location."
            )
        
        logger.info(f"EC2 SSH configuration validated: {ec2_config['key_name']}")
    
    def load_training_config(self) -> Dict[str, Any]:
        """Load training configuration with optional environment overrides."""
        training_config_file = self.config_dir / "training_config.json"
        with open(training_config_file, 'r') as f:
            config = json.load(f)
        
        # Optional environment variable overrides
        env_overrides = {
            'training.epochs': ('TRAINING_EPOCHS', int),
            'data.batch_size': ('TRAINING_BATCH_SIZE', int),
            'training.optimizer.lr': ('TRAINING_LEARNING_RATE', float),
        }
        
        for config_path, (env_var, converter) in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value:
                try:
                    converted_value = converter(env_value)
                    self._set_nested_config(config, config_path, converted_value)
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: {env_value}")
        
        return config
    
    def _set_nested_config(self, config: Dict[str, Any], path: str, value: Any):
        """Set a nested configuration value using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get_boto3_session_kwargs(self) -> Dict[str, str]:
        """
        Get kwargs for boto3.Session() initialization.
        
        Returns only explicitly configured values (region, profile).
        boto3 will use its default credential chain for authentication.
        """
        session_kwargs = {}
        
        if os.getenv('AWS_DEFAULT_REGION'):
            session_kwargs['region_name'] = os.getenv('AWS_DEFAULT_REGION')
        
        if os.getenv('AWS_PROFILE'):
            session_kwargs['profile_name'] = os.getenv('AWS_PROFILE')
        
        return session_kwargs
    
    def print_required_env_vars(self):
        """Print information about required configuration."""
        print("\n" + "="*70)
        print("Required Configuration")
        print("="*70)
        print("\n1. AWS Credentials (managed by AWS Toolkit/CLI):")
        print("   - Install AWS Toolkit in VS Code, OR")
        print("   - Run: aws configure")
        print("\n2. EC2 SSH Key (must be configured):")
        
        try:
            config = self.load_aws_config()
            ec2_config = config.get('ec2', {})
            
            key_name = ec2_config.get('key_name')
            key_path = ec2_config.get('key_path')
        except Exception:
            key_name = None
            key_path = None        
        
        if key_name is None or key_path is None:
            key_name = os.getenv('AWS_EC2_KEY_NAME')
            key_path = os.getenv('AWS_EC2_KEY_PATH')
            
            if key_name:
                print(f"   ✓ Key Name: {key_name}")
            else:
                print("   ✗ Key Name: Not set (required)")
                print("     Set in configs/aws_config.json or AWS_EC2_KEY_NAME env var")
            
            if key_path:
                key_exists = Path(key_path).exists()
                status = "✓" if key_exists else "✗"
                print(f"   {status} Key Path: {key_path}")
                if not key_exists:
                    print(f"     WARNING: File not found!")
            else:
                print("   ✗ Key Path: Not set (required)")
                print("     Set in configs/aws_config.json or AWS_EC2_KEY_PATH env var")
        
            print("\n" + "="*70)


def validate_aws_credentials() -> bool:
    """
    Validate that AWS credentials are accessible via boto3.
    
    Returns:
        True if credentials can be found, False otherwise
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        logger.info(f"AWS credentials validated for account: {identity['Account']}")
        logger.info(f"User ARN: {identity['Arn']}")
        return True
        
    except ImportError:
        logger.error("boto3 not installed. Run: pip install boto3")
        return False
    except Exception as e:
        # Catch all exceptions including NoCredentialsError
        error_msg = str(e)
        if "credentials" in error_msg.lower():
            logger.error("No AWS credentials found. Please configure AWS Toolkit or AWS CLI.")
        else:
            logger.error(f"AWS credential validation failed: {e}")
        return False


if __name__ == "__main__":
    """Test configuration loading"""
    logging.basicConfig(level=logging.INFO)
    
    loader = SecureConfigLoader()
    loader.print_required_env_vars()
    
    print("\nTesting AWS credential validation...")
    if validate_aws_credentials():
        print("✓ AWS credentials are valid and accessible")
    else:
        print("✗ AWS credentials not found or invalid")
        print("\nSetup Instructions:")
        print("1. Install AWS Toolkit extension in VS Code")
        print("2. Press Ctrl+Shift+P → 'AWS: Connect to AWS'")
        print("3. Configure EC2 keys in configs/aws_config.json")
