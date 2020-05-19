#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Function
torch.manual_seed(1)
import numpy as np
from torch.autograd import gradcheck


class TransposedConv(Function):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TransposedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = torch.nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size[0], kernel_size[1]))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.init_weights()
        print(self.kernel)

    def init_weights(self):
        nn.init.xavier_uniform_(self.kernel)
        nn.init.zeros_(self.bias)

    def transpose_kernel(self, size):
        out_size0 = (size[0] // self.kernel.shape[2] - 1) * self.kernel.shape[2] + size[0] % self.kernel.shape[2] + 1
        out_size1 = (size[1] // self.kernel.shape[3] - 1) * self.kernel.shape[3] + size[1] % self.kernel.shape[3] + 1
        transp_kern = torch.zeros((self.in_channels, self.out_channels, out_size0 * out_size1, size[0] * size[1]))
        print(size, out_size0, out_size1)
        #print(transp_kern)
        shift = 0
        # go by rows
        for i in range(out_size0 * out_size1):
            if i % out_size1 == 0:
                shift = (i // out_size1) * size[1]
            else:
                shift += 1
            # go by horizontal blocks
            for j in range(size[1]):
                if j < self.kernel.shape[2]:
                    transp_kern[:, :, i, shift+j * size[1]:shift+j * size[1]+self.kernel.shape[3]] = self.kernel[:, :, j]
        print(transp_kern)
        return torch.transpose(transp_kern, dim0=-2, dim1=-1)
    
    def forward(self, inp):
        self.inp = inp
        size = (inp.shape[2] + self.kernel.shape[2] - 1, inp.shape[3] + self.kernel.shape[3] - 1)
        print(size)
        transposed_kernel = self.transpose_kernel(size)
        #print(transposed_kernel)
        print(transposed_kernel.shape)
        res = torch.zeros((inp.shape[0], self.out_channels, transposed_kernel.shape[-2], 1))
        for b in range(inp.shape[0]):
            for i in range(self.out_channels):
                #buf = torch.zeros((transposed_kernel.shape[-2], 1))
                for j in range(self.in_channels):
                    res[b, i] += torch.mm(transposed_kernel[j, i], torch.flatten(inp[b, j]).view(transposed_kernel.shape[-1], 1))
                res[b, i] = torch.add(res[b, i], self.bias[i])
                #res[i] = buf
            #res = torch.mm(transposed_kernel, torch.flatten(inp))
        return res.reshape(inp.shape[0], self.out_channels, size[0], size[1])


class StaticTransposedConv(Function):
    @staticmethod
    def forward(ctx, inp, kernel, bias):
        '''
        Forward pass.
        Input:
            inp - torch.tensor, input image. Should be 4D tensor of shape
            [batch_size, in_channels, height, width]
            kernel - torch.tensor, kernel of convolution. Should be 4D tensor
            of shape [in_channels, out_channels, kernel_size0, kernel_size1]
            bias - torch.tensor, bias. Should be 1D tensor of shape [out_channels]
        Output:
            conv - torch.tensor, output image
        '''
        def transpose_kernel(size, kernel, in_channels, out_channels):
            '''
            Function for creating convolution matrix from kernel.
            Input:
                size - tuple, size of output image
                kernel - torch.tensor, layer kernel
                in_channels - int, number of channels in input image
                out_channels - int, number of channels in output image
            Output:
                transposed_kernel - torch.tensor, transposed convolution matrix
            '''
            out_size0 = (size[0] // kernel.shape[2] - 1) * kernel.shape[2] + size[0] % kernel.shape[2] + 1
            out_size1 = (size[1] // kernel.shape[3] - 1) * kernel.shape[3] + size[1] % kernel.shape[3] + 1
            transp_kern = torch.zeros((in_channels, out_channels, out_size0 * out_size1, size[0] * size[1]))
            shift = 0
            # go by rows
            for i in range(out_size0 * out_size1):
                if i % out_size1 == 0:
                    shift = (i // out_size1) * size[1]
                else:
                    shift += 1
                # go by horizontal blocks
                for j in range(size[1]):
                    if j < kernel.shape[2]:
                        transp_kern[:, :, i, shift+j * size[1]:shift+j * size[1]+kernel.shape[3]] = kernel[:, :, j]
            return torch.transpose(transp_kern, dim0=-2, dim1=-1)
        in_channels = kernel.shape[0]
        out_channels = kernel.shape[1]
        size = (inp.shape[2] + kernel.shape[2] - 1, inp.shape[3] + kernel.shape[3] - 1)
        # get transposed kernel
        transposed_kernel = transpose_kernel(size, kernel, in_channels, out_channels)
        ctx.save_for_backward(inp, transposed_kernel, bias, kernel)
        res = torch.zeros((inp.shape[0], out_channels, transposed_kernel.shape[-2], 1))
        for b in range(inp.shape[0]):
            for i in range(out_channels):
                for j in range(in_channels):
                    res[b, i] += torch.mm(transposed_kernel[j, i], torch.flatten(inp[b, j]).view(transposed_kernel.shape[-1], 1))
                res[b, i] = torch.add(res[b, i], bias[i])
        return res.reshape(inp.shape[0], out_channels, size[0], size[1])
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        Backward pass.
        Input:
            grad_output - torch.tensor, gradient from next layer
        Output:
            grad_input - torch.tensor, grad_output multiplied by gradient of
            forward's output wrt to input
            grad_kernel - torch.tensor, grad_output multiplied by gradient of
            forward's output wrt to kernel
            grad_bias - torch.tensor, grad_output multiplied by gradient of
            forward's output wrt to bias
        '''
        inp, transposed_kernel, bias, kernel = ctx.saved_tensors
        size = (inp.shape[2] + kernel.shape[2] - 1,
                inp.shape[3] + kernel.shape[3] - 1)
        out_size0 = (size[0] // kernel.shape[2] - 1) * kernel.shape[2] + size[0] % kernel.shape[2] + 1
        out_size1 = (size[1] // kernel.shape[3] - 1) * kernel.shape[3] + size[1] % kernel.shape[3] + 1
        in_channels = transposed_kernel.shape[0]
        out_channels = transposed_kernel.shape[1]
        grad_input = grad_kernel = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros((grad_output.shape[0], in_channels,
                                      out_size0, out_size1))
            for b in range(grad_output.shape[0]):
                for i in range(out_channels):
                    for j in range(in_channels):
                        buf = torch.mm(torch.flatten(grad_output[b, i]).view(1, size[0] * size[1]),
                                        transposed_kernel[j, i].float()).squeeze()
                        grad_input[b, j] += buf.view(out_size0, out_size1)
            #grad_input = torch.randn(1, 2, 2, 2, requires_grad=True)
        if ctx.needs_input_grad[1]:
            grad_kernel1 = torch.zeros((in_channels, out_channels, transposed_kernel.shape[-2],
                                        transposed_kernel.shape[-1]))
            for b in range(grad_output.shape[0]):
                for i in range(out_channels):
                    for j in range(in_channels):
                        grad_kernel1[j, i] += torch.mm(torch.flatten(grad_output[b, i]).view(transposed_kernel.shape[-2], 1),
                                                       torch.flatten(inp[b, j]).view(1, transposed_kernel.shape[-1]).float())
            # grad_kernel = grad_output.t().mm(inp)
            grad_kernel = torch.zeros((in_channels, out_channels,
                                       kernel.shape[-2], kernel.shape[-1]), dtype=torch.double)
            grad_kernel1 = torch.transpose(grad_kernel1, dim0=-1, dim1=-2)
            for j in range(grad_kernel1.shape[-2]):
                if j % out_size1 == 0:
                    shift = (j // out_size1) * size[1]
                else:
                    shift += 1
                for i in range(grad_kernel.shape[-2]):
                    grad_kernel[:, :, i] += grad_kernel1[:, :, j, shift+i*size[1]: shift+i*size[1]+kernel.shape[-1]]
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).sum(1).sum(1)
        return grad_input, grad_kernel, grad_bias


def test_transposed_conv():
    '''
    Function for testing transposed convolution forward pass.
    '''
    upsample = nn.ConvTranspose2d(2, 3, kernel_size=3, stride=1, padding=0)
    inp = torch.randn(1, 2, 2, 2)
    out = upsample(inp)
    tconv = StaticTransposedConv.apply
    buf = tconv(inp, upsample.weight, upsample.bias)
    assert torch.allclose(out, buf)
    print('PyTorch output', out)
    print('Custom layer output', buf)


def test_transposed_conv_back():
    '''
    Function for testing transposed convolution backward pass.
    '''
    upsample = nn.ConvTranspose2d(2, 3, kernel_size=3, stride=1, padding=0)
    inp = torch.randn(1, 2, 2, 2, requires_grad=True)
    inputs = (inp, torch.tensor(upsample.weight, dtype=torch.double, requires_grad=True),
              torch.tensor(upsample.bias, dtype=torch.double, requires_grad=True))
    tconv = StaticTransposedConv.apply
    test = gradcheck(tconv, inputs, eps=1e-3, atol=1e-3)
    if test:
        print('Backward works')
    else:
        print("Backward doesn't work")


def main():
    test_transposed_conv()
    test_transposed_conv_back()


if __name__ == '__main__':
    main()
