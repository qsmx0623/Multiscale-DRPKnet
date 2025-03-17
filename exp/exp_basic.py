import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
    '''
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    '''
    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                # 如果使用多个 GPU，创建一个以逗号分隔的设备 ID 字符串
                self.device = torch.device(f'cuda:{self.args.device_ids[0]}')  # 使用第一个 GPU 作为主设备
                print(f'Use multiple GPUs: {self.args.devices}')
            else:
                # 单 GPU 的情况
                self.device = torch.device(f'cuda:{self.args.gpu}')
                print(f'Use single GPU: cuda:{self.args.gpu}')
        else:
            self.device = torch.device('cpu')
            print('Use CPU')

        return self.device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
