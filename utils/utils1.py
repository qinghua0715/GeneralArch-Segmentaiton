import yaml
import argparse
import datetime
import sys
import os
from pathlib import Path


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def load_config(config_path: str):
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args(config_path: str):
    """解析命令行参数，支持配置文件默认值"""
    config = load_config(config_path)
    parser = argparse.ArgumentParser(description='Medical Image Segmentation')
    for key, value in config['experiment'].items():
        parser.add_argument(f'--{key}', default=value, type=type(value))
    return parser.parse_args()


def setup_logging(args):
    """设置日志记录"""
    log_dir = Path(args.save_path) / args.data / args.id
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"Log-{args.time}.txt"
    sys.stdout = Logger(logfile)


# 指标累计器
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        # self.reset()
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count