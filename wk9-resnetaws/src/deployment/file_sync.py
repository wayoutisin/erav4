#!/usr/bin/env python3
"""
File synchronization utilities for AWS deployment
"""
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from ..utils.logger_utils import setup_logger


class FileSync:
    def __init__(self, aws_config_path: str):
        """Initialize file sync utility"""
        import json
        with open(aws_config_path, 'r') as f:
            self.aws_config = json.load(f)
        
        self.deployment_config = self.aws_config['deployment']
        self.ec2_config = self.aws_config['ec2']
        self.logger = setup_logger('file_sync')
    
    def sync_to_remote(self, remote_ip: str, key_name: str) -> bool:
        """Sync local project to remote instance"""
        # Get full key path from config
        key_path = self.ec2_config.get('key_path')
        if not key_path:
            self.logger.error("Key path not configured")
            return False
        
        local_project_root = Path.cwd()
        remote_workspace = self.deployment_config['remote_workspace']
        
        # Create remote workspace directory
        ssh_cmd = [
            'ssh', '-i', key_path,
            '-o', 'StrictHostKeyChecking=no',
            f'ubuntu@{remote_ip}',
            f'mkdir -p {remote_workspace}'
        ]
        
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Failed to create remote directory: {result.stderr}")
            return False
        
        # Sync project files
        return self.sync_directory(
            str(local_project_root),
            f'{remote_ip}:{remote_workspace}',
            key_path,
            exclude_patterns=self.deployment_config.get('sync_exclude', [])
        )
    
    def sync_directory(self, 
                      source: str, 
                      destination: str, 
                      key_path: str,
                      exclude_patterns: Optional[List[str]] = None,
                      direction: str = 'to_remote') -> bool:
        """Sync directory using rsync"""
        
        exclude_patterns = exclude_patterns or []
        
        # Build rsync command
        rsync_cmd = [
            'rsync', '-avz', '--progress',
            '-e', f'ssh -i {key_path} -o StrictHostKeyChecking=no'
        ]
        
        # Add exclude patterns
        for pattern in exclude_patterns:
            rsync_cmd.extend(['--exclude', pattern])
        
        # Add source and destination
        if direction == 'to_remote':
            rsync_cmd.extend([f'{source}/', f'ubuntu@{destination}/'])
        else:  # from_remote
            rsync_cmd.extend([f'ubuntu@{source}/', f'{destination}/'])
        
        self.logger.info(f"Syncing: {source} -> {destination}")
        self.logger.debug(f"Command: {' '.join(rsync_cmd)}")
        
        try:
            result = subprocess.run(rsync_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Sync completed successfully")
                return True
            else:
                self.logger.error(f"Sync failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Sync error: {str(e)}")
            return False
    
    def upload_file(self, 
                   local_file: str, 
                   remote_ip: str, 
                   remote_path: str,
                   key_name: str) -> bool:
        """Upload single file to remote instance"""
        # Get full key path from config
        key_path = self.ec2_config.get('key_path')
        if not key_path:
            self.logger.error("Key path not configured")
            return False
        
        scp_cmd = [
            'scp', '-i', key_path,
            '-o', 'StrictHostKeyChecking=no',
            local_file,
            f'ubuntu@{remote_ip}:{remote_path}'
        ]
        
        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"File uploaded: {local_file} -> {remote_path}")
                return True
            else:
                self.logger.error(f"Upload failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Upload error: {str(e)}")
            return False
    
    def download_file(self, 
                     remote_ip: str, 
                     remote_path: str,
                     local_file: str,
                     key_name: str) -> bool:
        """Download single file from remote instance"""
        # Get full key path from config
        key_path = self.ec2_config.get('key_path')
        if not key_path:
            self.logger.error("Key path not configured")
            return False
        
        scp_cmd = [
            'scp', '-i', key_path,
            '-o', 'StrictHostKeyChecking=no',
            f'ubuntu@{remote_ip}:{remote_path}',
            local_file
        ]
        
        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"File downloaded: {remote_path} -> {local_file}")
                return True
            else:
                self.logger.error(f"Download failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Download error: {str(e)}")
            return False
    
    def sync_s3_data(self, s3_path: str, local_path: str, direction: str = 'download') -> bool:
        """Sync data with S3 bucket"""
        if not os.path.exists(local_path) and direction == 'upload':
            self.logger.error(f"Local path does not exist: {local_path}")
            return False
        
        if direction == 'download':
            aws_cmd = ['aws', 's3', 'sync', s3_path, local_path]
        else:  # upload
            aws_cmd = ['aws', 's3', 'sync', local_path, s3_path]
        
        try:
            result = subprocess.run(aws_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"S3 sync completed: {s3_path} <-> {local_path}")
                return True
            else:
                self.logger.error(f"S3 sync failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"S3 sync error: {str(e)}")
            return False