#!/usr/bin/env python3
"""
AWS EC2 Instance Manager for ML Training
"""
import json
import time
import boto3
import argparse
from pathlib import Path
from botocore.exceptions import ClientError, BotoCoreError

from ..utils.logger_utils import setup_logger
from ..utils.config_utils import SecureConfigLoader


class EC2Manager:
    def __init__(self, config=None):
        """
        Initialize EC2 Manager using boto3's default credential chain.
        
        AWS credentials are automatically discovered from:
        - AWS Toolkit for VS Code
        - AWS CLI profiles
        - Environment variables
        - IAM roles (when running on EC2)
        """
        
        if config is None or isinstance(config, str):
            # Load configuration (uses boto3 default credential chain)
            config_loader = SecureConfigLoader()
            self.config = config_loader.load_aws_config()
            session_kwargs = config_loader.get_boto3_session_kwargs()
        else:
            self.config = config
            session_kwargs = {}
            if self.config.get('aws', {}).get('region'):
                session_kwargs['region_name'] = self.config['aws']['region']
        
        self.aws_config = self.config['aws']
        self.ec2_config = self.config['ec2']
        
        # Create boto3 session using default credential chain
        session = boto3.Session(**session_kwargs)
        
        self.ec2_client = session.client('ec2')
        self.ec2_resource = session.resource('ec2')
        self.ssm_client = session.client('ssm')
        
        self.logger = setup_logger('ec2_manager')
        self.instance_id = None
    
    def create_instance(self):
        """Create and launch EC2 instance"""
        try:
            # Prepare user data script for instance initialization
            user_data_script = self._generate_user_data()
            
            # Launch instance
            response = self.ec2_client.run_instances(
                ImageId=self.ec2_config['ami_id'],
                MinCount=1,
                MaxCount=1,
                InstanceType=self.ec2_config['instance_type'],
                KeyName=self.ec2_config['key_name'],
                SecurityGroups=self.ec2_config['security_groups'],
                SubnetId=self.ec2_config.get('subnet_id'),
                IamInstanceProfile={
                    'Name': self.ec2_config.get('iam_instance_profile', 'EC2-ML-Role')
                },
                BlockDeviceMappings=[
                    {
                        'DeviceName': '/dev/sda1',
                        'Ebs': {
                            'VolumeSize': self.ec2_config['storage']['volume_size'],
                            'VolumeType': self.ec2_config['storage']['volume_type'],
                            'DeleteOnTermination': True
                        }
                    }
                ],
                UserData=user_data_script,
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': k, 'Value': v} 
                            for k, v in self.ec2_config['tags'].items()
                        ]
                    }
                ]
            )
            
            self.instance_id = response['Instances'][0]['InstanceId']
            self.logger.info(f"EC2 instance created: {self.instance_id}")
            
            # Wait for instance to be running
            self.logger.info("Waiting for instance to be running...")
            waiter = self.ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[self.instance_id])
            
            # Get instance details
            instance_info = self.get_instance_info()
            self.logger.info(f"Instance is running. Public IP: {instance_info['public_ip']}")
            
            return self.instance_id, instance_info
            
        except ClientError as e:
            self.logger.error(f"Failed to create instance: {e}")
            raise
    
    def terminate_instance(self, instance_id=None):
        """Terminate EC2 instance"""
        target_id = instance_id or self.instance_id
        if not target_id:
            self.logger.error("No instance ID provided")
            return False
        
        try:
            self.ec2_client.terminate_instances(InstanceIds=[target_id])
            self.logger.info(f"Instance {target_id} termination initiated")
            return True
        except ClientError as e:
            self.logger.error(f"Failed to terminate instance: {e}")
            return False
    
    def get_instance_info(self, instance_id=None):
        """Get instance information"""
        target_id = instance_id or self.instance_id
        if not target_id:
            return None
        
        try:
            response = self.ec2_client.describe_instances(InstanceIds=[target_id])
            instance = response['Reservations'][0]['Instances'][0]
            
            return {
                'instance_id': instance['InstanceId'],
                'state': instance['State']['Name'],
                'instance_type': instance['InstanceType'],
                'public_ip': instance.get('PublicIpAddress'),
                'private_ip': instance.get('PrivateIpAddress'),
                'launch_time': instance['LaunchTime'],
                'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
            }
        except ClientError as e:
            self.logger.error(f"Failed to get instance info: {e}")
            return None
    
    def wait_for_instance_ready(self, instance_id=None, timeout=300):
        """Wait for instance to be ready for SSH connection"""
        target_id = instance_id or self.instance_id
        if not target_id:
            return False
        
        self.logger.info("Waiting for instance to be ready for connections...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if instance is running
                info = self.get_instance_info(target_id)
                if info and info['state'] == 'running':
                    # Try to check system status
                    status_response = self.ec2_client.describe_instance_status(
                        InstanceIds=[target_id]
                    )
                    
                    if status_response['InstanceStatuses']:
                        status = status_response['InstanceStatuses'][0]
                        if (status['InstanceStatus']['Status'] == 'ok' and 
                            status['SystemStatus']['Status'] == 'ok'):
                            self.logger.info("Instance is ready!")
                            return True
                
                time.sleep(10)
            except ClientError:
                time.sleep(10)
                continue
        
        self.logger.error("Timeout waiting for instance to be ready")
        return False
    
    def _generate_user_data(self):
        """Generate user data script for instance initialization"""
        script = f"""#!/bin/bash
# Update system
apt-get update -y

# Install required packages
apt-get install -y python3-pip python3-venv git htop nvtop

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Install Docker (optional)
apt-get install -y docker.io
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# Create workspace directory
mkdir -p {self.config['deployment']['remote_workspace']}
chown ubuntu:ubuntu {self.config['deployment']['remote_workspace']}

# Install Conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /home/ubuntu/miniconda3
chown -R ubuntu:ubuntu /home/ubuntu/miniconda3

# Initialize conda for ubuntu user
sudo -u ubuntu /home/ubuntu/miniconda3/bin/conda init bash

# Setup CloudWatch agent (if enabled)
{self._generate_cloudwatch_config() if self.config.get('monitoring', {}).get('enable_cloudwatch') else ''}

# Signal completion
echo "Instance setup completed" > /tmp/setup_complete.log
"""
        return script
    
    def _generate_cloudwatch_config(self):
        """Generate CloudWatch agent configuration"""
        if not self.config.get('monitoring', {}).get('enable_cloudwatch'):
            return ""
        
        return f"""
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
dpkg -i amazon-cloudwatch-agent.rpm

# Configure CloudWatch agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{{
    "agent": {{
        "metrics_collection_interval": 60,
        "run_as_user": "cwagent"
    }},
    "logs": {{
        "logs_collected": {{
            "files": {{
                "collect_list": [
                    {{
                        "file_path": "/var/log/cloud-init-output.log",
                        "log_group_name": "{self.config['monitoring']['log_group']}",
                        "log_stream_name": "{{instance_id}}/cloud-init-output.log"
                    }},
                    {{
                        "file_path": "/home/ubuntu/resnet-training/logs/*.log",
                        "log_group_name": "{self.config['monitoring']['log_group']}",
                        "log_stream_name": "{{instance_id}}/training.log"
                    }}
                ]
            }}
        }}
    }},
    "metrics": {{
        "namespace": "{self.config['monitoring']['metrics_namespace']}",
        "metrics_collected": {{
            "cpu": {{
                "measurement": ["cpu_usage_idle", "cpu_usage_iowait", "cpu_usage_user", "cpu_usage_system"],
                "metrics_collection_interval": 60
            }},
            "disk": {{
                "measurement": ["used_percent"],
                "metrics_collection_interval": 60,
                "resources": ["*"]
            }},
            "mem": {{
                "measurement": ["mem_used_percent"],
                "metrics_collection_interval": 60
            }}
        }}
    }}
}}
EOF

# Start CloudWatch agent
systemctl enable amazon-cloudwatch-agent
systemctl start amazon-cloudwatch-agent
"""


def main():
    parser = argparse.ArgumentParser(description='Manage EC2 instances for ML training')
    parser.add_argument('--config', type=str, required=True, help='AWS configuration file')
    parser.add_argument('--action', type=str, choices=['create', 'terminate', 'info'], 
                       required=True, help='Action to perform')
    parser.add_argument('--instance-id', type=str, help='Instance ID for terminate/info actions')
    
    args = parser.parse_args()
    
    manager = EC2Manager(args.config)
    
    if args.action == 'create':
        instance_id, info = manager.create_instance()
        print(f"Instance created: {instance_id}")
        print(f"Public IP: {info['public_ip']}")
        
        # Wait for instance to be ready
        if manager.wait_for_instance_ready():
            print("Instance is ready for connections!")
        else:
            print("Instance may not be fully ready. Please check manually.")
    
    elif args.action == 'terminate':
        instance_id = args.instance_id or manager.instance_id
        if manager.terminate_instance(instance_id):
            print(f"Instance {instance_id} termination initiated")
        else:
            print("Failed to terminate instance")
    
    elif args.action == 'info':
        instance_id = args.instance_id or manager.instance_id
        info = manager.get_instance_info(instance_id)
        if info:
            print(json.dumps(info, indent=2, default=str))
        else:
            print("Could not retrieve instance information")


if __name__ == '__main__':
    main()