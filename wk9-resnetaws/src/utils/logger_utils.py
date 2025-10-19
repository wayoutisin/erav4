"""
Logging utilities for ResNet18 training project
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(name: str, 
                log_level: str = 'INFO',
                log_file: Optional[str] = None,
                console_output: bool = True) -> logging.Logger:
    """Setup and configure logger"""
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_training_logger(output_dir: str, 
                         experiment_name: Optional[str] = None) -> logging.Logger:
    """Create logger specifically for training experiments"""
    
    if experiment_name is None:
        experiment_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create logs directory
    logs_dir = Path(output_dir) / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file path
    log_file = logs_dir / f"{experiment_name}.log"
    
    # Setup logger
    logger = setup_logger(
        name='training',
        log_level='INFO',
        log_file=str(log_file),
        console_output=True
    )
    
    logger.info(f"Training logger initialized. Log file: {log_file}")
    
    return logger


class TrainingLogger:
    """Enhanced training logger with metrics tracking"""
    
    def __init__(self, output_dir: str, experiment_name: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directories
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = create_training_logger(str(self.output_dir), self.experiment_name)
        
        # Metrics tracking
        self.metrics_file = self.logs_dir / f"{self.experiment_name}_metrics.csv"
        self._initialize_metrics_file()
    
    def _initialize_metrics_file(self):
        """Initialize CSV file for metrics logging"""
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,timestamp\n")
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float, lr: float):
        """Log epoch results"""
        timestamp = datetime.now().isoformat()
        
        # Log to console/file
        self.logger.info(
            f"Epoch {epoch:3d} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"LR: {lr:.6f}"
        )
        
        # Log to CSV
        with open(self.metrics_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                   f"{val_loss:.6f},{val_acc:.6f},{lr:.8f},{timestamp}\n")
    
    def log_best_model(self, epoch: int, val_acc: float, model_path: str):
        """Log best model information"""
        self.logger.info(f"New best model at epoch {epoch}! "
                        f"Validation accuracy: {val_acc:.4f}, "
                        f"Saved to: {model_path}")
    
    def log_training_start(self, config: dict, model_info: dict):
        """Log training start information"""
        self.logger.info("="*80)
        self.logger.info("TRAINING STARTED")
        self.logger.info("="*80)
        
        # Log configuration
        self.logger.info("Configuration:")
        for section, params in config.items():
            self.logger.info(f"  {section}: {params}")
        
        # Log model information
        self.logger.info("Model Information:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("="*80)
    
    def log_training_end(self, best_acc: float, total_time: float):
        """Log training completion information"""
        self.logger.info("="*80)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"Best validation accuracy: {best_acc:.4f}")
        self.logger.info(f"Total training time: {total_time:.2f} seconds")
        self.logger.info(f"Metrics saved to: {self.metrics_file}")
        self.logger.info("="*80)
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Log error information"""
        self.logger.error(f"Training error: {error_msg}")
        if exception:
            self.logger.exception("Exception details:")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)


def log_system_info(logger: logging.Logger):
    """Log system information"""
    import torch
    import platform
    import psutil
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python Version: {platform.python_version()}")
    logger.info(f"  PyTorch Version: {torch.__version__}")
    
    # CPU info
    logger.info(f"  CPU: {platform.processor()}")
    logger.info(f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"  RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    # GPU info
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
    else:
        logger.info("  GPU: Not available")


def setup_tensorboard_logger(log_dir: str, experiment_name: str):
    """Setup TensorBoard logging (requires tensorboard package)"""
    try:
        from torch.utils.tensorboard import SummaryWriter
        
        tb_log_dir = Path(log_dir) / 'tensorboard' / experiment_name
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        return writer
    except ImportError:
        logger = setup_logger('tensorboard')
        logger.warning("TensorBoard not available. Install with 'pip install tensorboard' for TensorBoard logging")
        return None


class TensorBoardLogger:
    """TensorBoard logging wrapper"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.writer = setup_tensorboard_logger(log_dir, experiment_name)
        self.enabled = self.writer is not None
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: dict, step: int):
        """Log multiple scalar values"""
        if self.enabled:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram of values"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log image"""
        if self.enabled:
            self.writer.add_image(tag, image, step)
    
    def log_graph(self, model, input_tensor):
        """Log model graph"""
        if self.enabled:
            self.writer.add_graph(model, input_tensor)
    
    def close(self):
        """Close writer"""
        if self.enabled:
            self.writer.close()