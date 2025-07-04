import logging
from datetime import datetime
import os

class TrainingLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_logger()
    
    def _setup_logger(self):
        # 创建日志目录
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)  # 关键修复
        
        log_file = os.path.join(
            log_dir,
            f"train_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
        )
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def log_metrics(self, epoch: int, loss: float):
        self.logger.info(f"Epoch {epoch} | Loss: {loss:.4f}")
    
    def log_error(self, error: Exception):
        self.logger.error(f"Error occurred: {str(error)}", exc_info=True)