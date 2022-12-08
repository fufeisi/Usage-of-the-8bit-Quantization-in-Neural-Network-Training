import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
from torch.nn.modules.utils import _pair
import quantization.tool as tl

quan = True

class qlinear(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias):
    if quan and tl.my_quantize.train:
      qx = tl.my_quantize.forward(x)
      ctx.save_for_backward(qx, weight)
    else:
      ctx.save_for_backward(x, weight)
    out = F.linear(x, weight, bias)  # (m, k) x (k, n)
    return out
  
  @staticmethod
  def backward(ctx, dout):
    if quan and tl.my_quantize.train:
      qx, weight = ctx.saved_tensors
      x = qx.dequantize()
    else:
      x, weight = ctx.saved_tensors
    dx, dweight = None, None
    if ctx.needs_input_grad[0]:
      dx = F.linear(dout, weight.T)
    if ctx.needs_input_grad[1]:
      dim = list(range(len(x.shape)-1))
      dweight = torch.tensordot(dout, x, dims=(dim, dim))
    return dx, dweight, dout

class qrelu(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    out = torch.relu(x)
    if quan and tl.my_quantize.train:
      qout = tl.my_quantize.forward(out)
      ctx.save_for_backward(qout)
    else:
      ctx.save_for_backward(out)
    return out
  
  @staticmethod
  def backward(ctx, dout):
    if quan and tl.my_quantize.train:
      qout = ctx.saved_tensors[0]
      out = qout.dequantize()
    else:
      out = ctx.saved_tensors[0]
    dx = torch.mul(dout, torch.sign(out))
    return dx

class qconv2d(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, module=None):
    if quan and tl.my_quantize.train:
      qx = tl.my_quantize.forward(x)
      ctx.save_for_backward(qx, weight)
    else:
      ctx.save_for_backward(x, weight)
    ctx.bias_sizes_opt = 0 if bias is None else bias.shape[0]
    ctx.module = module
    ctx.stride = stride

    ctx.padding = padding
    ctx.dilation = dilation
    ctx.groups = groups

    return F.conv2d(input=x, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

  @staticmethod
  def backward(ctx, grad_output):
    if quan and tl.my_quantize.train:
      qx, weight = ctx.saved_tensors
      x = qx.dequantize()
    else:
      x, weight = ctx.saved_tensors
    bias_sizes_opt = ctx.bias_sizes_opt
    stride = ctx.stride
    padding = ctx.padding
    dilation = ctx.dilation
    groups = ctx.groups

    grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(grad_output, x, weight, _pair(bias_sizes_opt),
                                               _pair(stride), _pair(padding), _pair(dilation),
                                               False, [0], groups, ctx.needs_input_grad[:3])

    return grad_input, grad_weight, grad_bias, None, None, None, None, None

class qLinear(torch.nn.Linear):
  def forward(self, input: Tensor) -> Tensor:
      return qlinear.apply(input, self.weight, self.bias)

class qReLu(torch.nn.ReLU):
  def forward(self, input: Tensor) -> Tensor:
    return qrelu.apply(input)

class qConv2d(torch.nn.Conv2d):
  def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
      if self.padding_mode != 'zeros':
          return qconv2d.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                          weight, bias, self.stride,
                          _pair(0), self.dilation, self.groups)
      return qconv2d.apply(input, weight, bias, self.stride,
                      self.padding, self.dilation, self.groups)
  def forward(self, input: Tensor) -> Tensor:
    return self._conv_forward(input, self.weight, self.bias)
