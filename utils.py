import os
import torch
import logging
import shutil
import gpustat
import random


class AverageMeter:
    """Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

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


def accuracy(output, target, topk=(1, )):
    """Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.float().topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_logging(log_file):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def setup_gpus():
    """Adapted from https://github.com/bamos/setGPU/blob/master/setGPU.py
    """
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)
    best_gpu = min(pairs, key=lambda x: x[1])[0]
    return best_gpu


def save_checkpoint(state, is_best, path, name='model_latest.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = path + '/' + name
    torch.save(state, save_path)
    logging.info('checkpoint saved to {}'.format(save_path))
    if is_best:
        shutil.copyfile(save_path, path + '/model_best.pth.tar')