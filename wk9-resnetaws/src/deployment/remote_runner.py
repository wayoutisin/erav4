#!/usr/bin/env python3
"""
Remote command execution utilities for AWS deployment
"""
import subprocess
import time
from typing import Optional, Dict, Any

from ..utils.logger_utils import setup_logger


class RemoteRunner:
    def __init__(self, remote_ip: str, key_path: str, aws_config: Dict[str, Any]):
        """Initialize remote runner"""
        self.remote_ip = remote_ip
        self.key_path = key_path
        self.aws_config = aws_config
        self.logger = setup_logger('remote_runner')
        
        self.ssh_base_cmd = [
            'ssh', '-i', f'{key_path}',
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'ConnectTimeout=30',
            f'ubuntu@{remote_ip}'
        ]
    
    def execute_command(self, 
                       command: str, 
                       background: bool = False,
                       wait_for_completion: bool = True,
                       timeout: Optional[int] = None) -> Optional[str]:
        """Execute command on remote instance"""
        
        if background:
            # For background processes, use nohup and redirect output
            command = f'nohup {command} > /tmp/command_output.log 2>&1 & echo $!'
        
        ssh_cmd = self.ssh_base_cmd + [command]
        print(ssh_cmd);
        self.logger.info(f"Executing: {command}")
        self.logger.debug(f"SSH command: {' '.join(ssh_cmd)}")
        
        try:
            if wait_for_completion:
                result = subprocess.run(
                    ssh_cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    self.logger.info("Command executed successfully")
                    if background:
                        # Return process ID for background jobs
                        return result.stdout.strip()
                    return result.stdout
                else:
                    self.logger.error(f"Command failed: {result.stderr}")
                    return None
            else:
                # Start process without waiting
                process = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE, text=True)
                return str(process.pid)
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timeout after {timeout} seconds")
            return None
        except Exception as e:
            self.logger.error(f"Command execution error: {str(e)}")
            return None
    
    def check_process_status(self, pid: str) -> Dict[str, Any]:
        """Check if a background process is still running"""
        check_cmd = f'ps -p {pid} -o pid,ppid,cmd --no-headers'
        
        result = self.execute_command(check_cmd, wait_for_completion=True)
        
        if result and result.strip():
            parts = result.strip().split(None, 2)
            return {
                'running': True,
                'pid': parts[0],
                'ppid': parts[1],
                'command': parts[2] if len(parts) > 2 else 'N/A'
            }
        else:
            return {'running': False}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information from remote instance"""
        commands = {
            'cpu_info': 'lscpu | grep "Model name" | head -1',
            'memory_info': 'free -h | grep "Mem:" | awk \'{print $2}\'',
            'disk_info': 'df -h / | tail -1 | awk \'{print $2, $3, $4, $5}\'',
            'gpu_info': 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "No GPU"',
            'uptime': 'uptime',
            'load_average': 'cat /proc/loadavg',
            'python_version': 'python3 --version',
            'conda_envs': 'source ~/.bashrc && conda env list'
        }
        
        system_info = {}
        for key, cmd in commands.items():
            result = self.execute_command(cmd, wait_for_completion=True)
            system_info[key] = result.strip() if result else 'N/A'
        
        return system_info
    
    def monitor_gpu_usage(self, duration: int = 60) -> bool:
        """Monitor GPU usage for specified duration"""
        monitor_cmd = f'''
        for i in {{1..{duration}}}; do
            echo "=== $(date) ==="
            nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
            sleep 1
        done
        '''
        
        result = self.execute_command(monitor_cmd, wait_for_completion=True, timeout=duration+10)
        return result is not None
    
    def tail_log_file(self, log_file: str, lines: int = 50) -> Optional[str]:
        """Get last N lines from a log file"""
        tail_cmd = f'tail -n {lines} {log_file}'
        return self.execute_command(tail_cmd, wait_for_completion=True)
    
    def check_training_progress(self, log_file: str) -> Dict[str, Any]:
        """Parse training log to get progress information"""
        # Get last 100 lines of log
        log_content = self.tail_log_file(log_file, 100)
        
        if not log_content:
            return {'error': 'Could not read log file'}
        
        progress_info = {
            'current_epoch': None,
            'total_epochs': None,
            'latest_train_loss': None,
            'latest_val_loss': None,
            'latest_train_acc': None,
            'latest_val_acc': None,
            'last_update': None
        }
        
        lines = log_content.split('\n')
        
        # Parse log lines (this is specific to your training script format)
        for line in reversed(lines):
            if 'Epoch' in line and 'Train Loss' in line:
                # Example: "Epoch 5/50 - Train Loss: 0.1234, Train Acc: 0.8765, Val Loss: 0.2345, Val Acc: 0.7654"
                parts = line.split(' - ')
                if len(parts) >= 2:
                    epoch_part = parts[0].split('/')
                    if len(epoch_part) == 2:
                        try:
                            progress_info['current_epoch'] = int(epoch_part[0].split()[-1])
                            progress_info['total_epochs'] = int(epoch_part[1])
                        except ValueError:
                            pass
                    
                    # Parse metrics
                    metrics_part = parts[1]
                    if 'Train Loss:' in metrics_part:
                        try:
                            train_loss = float(metrics_part.split('Train Loss:')[1].split(',')[0].strip())
                            progress_info['latest_train_loss'] = train_loss
                        except (ValueError, IndexError):
                            pass
                    
                    if 'Val Loss:' in metrics_part:
                        try:
                            val_loss = float(metrics_part.split('Val Loss:')[1].split(',')[0].strip())
                            progress_info['latest_val_loss'] = val_loss
                        except (ValueError, IndexError):
                            pass
                    
                    if 'Train Acc:' in metrics_part:
                        try:
                            train_acc = float(metrics_part.split('Train Acc:')[1].split(',')[0].strip())
                            progress_info['latest_train_acc'] = train_acc
                        except (ValueError, IndexError):
                            pass
                    
                    if 'Val Acc:' in metrics_part:
                        try:
                            val_acc = float(metrics_part.split('Val Acc:')[1].split(',')[0].strip())
                            progress_info['latest_val_acc'] = val_acc
                        except (ValueError, IndexError):
                            pass
                
                break
        
        # Get timestamp of last log entry
        if lines:
            progress_info['last_update'] = lines[-1][:19] if len(lines[-1]) > 19 else 'Unknown'
        
        return progress_info
    
    def cleanup_remote_files(self, directories: list) -> bool:
        """Clean up specified directories on remote instance"""
        for directory in directories:
            cmd = f'rm -rf {directory}'
            result = self.execute_command(cmd, wait_for_completion=True)
            if result is None:
                self.logger.error(f"Failed to cleanup directory: {directory}")
                return False
        
        self.logger.info("Remote cleanup completed")
        return True