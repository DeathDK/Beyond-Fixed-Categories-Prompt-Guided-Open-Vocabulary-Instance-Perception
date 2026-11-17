# @Time    : 2024/3/1 11:17
# @Author  : zhangchenming
import time
import torch
import argparse
import sys
import thop
from easydict import EasyDict
from tqdm import tqdm

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    args.run_mode = 'measure'
    return args, cfgs


def main():
    args, cfgs = parse_config()
    model = build_trainer(args, cfgs, local_rank=0, global_rank=0, logger=None, tb_writer=None).model

    shape = [1, 3, 544, 960]
    infer_time(model, shape)
    measure(model, shape)


@torch.no_grad()
def measure(model, shape):
    model.eval()

    inputs = {'left': torch.randn(shape).cuda(),
              'right': torch.randn(shape).cuda()}

    flops, params = thop.profile(model, inputs=(inputs,))
    print("Number of calculates:%.2fGFlops" % (flops / 1e9))
    print("Number of parameters:%.2fM" % (params / 1e6))


@torch.no_grad()
def infer_time(model, shape):
    model.eval()
    repetitions = 100

    inputs = {'left': torch.randn(shape).cuda(),
              'right': torch.randn(shape).cuda()}

    # Warm-up: GPUs may be idle for power saving, so warm up first
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)

    # synchronize waits until all GPU work finishes before returning to the CPU
    # torch.cuda.synchronize()

    # Set up CUDA Events for timing (PyTorch-recommended, most reliable)
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Initialize a timing buffer
    # timings = np.zeros((repetitions, 1))

    all_time = 0
    print('testing ...\n')
    with torch.no_grad():
        for _ in tqdm(range(repetitions)):
            # starter.record()
            infer_start = time.perf_counter()
            # infer_start = time.time()
            result = model(inputs)
            print(result.keys())
            # ender.record()
            all_time += time.perf_counter() - infer_start
            # torch.cuda.synchronize()  # wait for GPU work to finish

            # curr_time = starter.elapsed_time(ender)  # elapsed ms from starter to ender
            # timings[rep] = curr_time

    # avg = timings.sum() / repetitions
    # print('\navg_time=%.3fms\n' % avg)
    print(all_time / repetitions * 1000)


if __name__ == '__main__':
    main()
