from cmath import inf
import torch, time
import subprocess as sp
from threading import Timer
from torch.ao.ns.fx.utils import compute_sqnr

class QuanNames(object):
     def __init__(self) -> None:
          self.used = 0
          self.l = 32
          self.names = [self.l-i for i in range(self.l)]
          self.print = 1
     def reset(self):
          self.used = 0
          self.names = [self.l-i for i in range(self.l)]
     def extend(self):
          self.l = 2*self.l
          self.names = [self.l-i for i in range(self.l//2)]
     def get(self):
          self.used += 1
          if self.used > self.l:
               self.extend()
          return self.names.pop()

def sum_memory(prof):
     res = 0
     for item in prof.key_averages():
          if 'self_cuda_memory_usage' in dir(item):
               res +=item.self_cuda_memory_usage
     return res

def tensor_size(x):
     res = 1
     for i in x.shape:
          res *= i
     return res

def main_log(file_name, content):
     with open(file_name, 'a') as f:
          f.write(content+'\n')

class QuantizeBuffer:
     def __init__(self):
          self.buffer_dict = {}
          self.is_count = False
          self.count = 0
          self.names = QuanNames()
          self.train = True
          self.sqnr = {}
          self.count_time = 0
     def forward(self, x):
          if 'quan_name' not in dir(x):
               x.quan_name = self.names.get()
               self.buffer_dict[x.quan_name] = torch.quantize_per_tensor_dynamic(x, torch.quint8, False)
               if self.is_count:
                    start_time = time.time()
                    self.count += tensor_size(x)
                    self.sqnr[x.quan_name] = compute_sqnr(x, self.buffer_dict[x.quan_name].dequantize()).item()
                    self.count_time = time.time() - start_time
          return self.buffer_dict[x.quan_name]
     def start_train(self):
          self.train = True
          self.buffer_dict = {}
          self.names.reset()
     def stop_train(self):
          self.train = False
          self.buffer_dict = {}
          self.names.reset()
     def reset_memory(self):
          self.buffer_dict = {}
          self.names.reset()
     def start_count(self):
          self.sqnr = {}
          self.count = 0
          self.is_count = True
          self.count_time = 0
     def stop_count(self):
          self.sqnr = {}
          self.count = 0
          self.is_count = False
     def report(self):
          print(f'Total tensor we quantized {self.count/(250*1000)} MB; SQNR: {self.sqnr};')

my_quantize = QuantizeBuffer()


def get_gpu_info():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
    try:
        gpu_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    gpu_use_values = sum([int(item.split()[0]) for item in gpu_use_info])/len(gpu_use_info)
    return gpu_use_values

class PrintGPU:
    def __init__(self) -> None:
        self.is_print = True
        self.values = []
    def print_gpu(self, n):
        if self.is_print:
            Timer(n, lambda :self.print_gpu(n)).start()
        self.values.append(get_gpu_info())
    def cancel(self):
        self.is_print= False
    def report(self):
        if len(self.values) >= 1:
            print(f'Avg GPU usage: {sum(self.values)/len(self.values)}.')

if __name__ == '__main__':
    A = PrintGPU()
    A.print_gpu(5)
    time.sleep(20)
    A.cancel()

