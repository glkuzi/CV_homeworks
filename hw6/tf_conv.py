#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf

tf.keras.backend.clear_session()  # For easy reset of notebook state.
from tensorflow.keras import layers
import numpy as np
SEED = 21
np.random.seed(SEED)
tf.random.set_seed(SEED)


class TransposedConvTF(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TransposedConvTF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def build(self):
        self.kernel = self.add_weight(shape=(self.in_channels,
                                             self.out_channels,
                                             self.kernel_size[0],
                                             self.kernel_size[1]),
                                      initializer='random_normal',
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.out_channels),
                                    initializer='zeros',
                                    trainable=True)
    
    def transpose_kernel(self):
        '''
        Function for creating convolution matrix from kernel.
        Output:
            transposed_kernel - tf.tensor, transposed convolution matrix
        '''
        out_size0 = (self.size[0] // self.kernel.shape[2] - 1) * self.kernel.shape[2] + self.size[0] % self.kernel.shape[2] + 1
        out_size1 = (self.size[1] // self.kernel.shape[3] - 1) * self.kernel.shape[3] + self.size[1] % self.kernel.shape[3] + 1
        shift = 0
        buf_matrices = []
        for i in range(out_size0 * out_size1):
            buf_matrix = tf.Variable(tf.zeros((self.size[0] * self.size[1], self.kernel.shape[2] * self.kernel.shape[3])))
            if i % out_size1 == 0:
                shift = (i // out_size1) * self.size[1]
            else:
                shift += 1
            local_shift = 0
            for j in range(buf_matrix.shape[-1]):
                if j % self.kernel.shape[-1] == 0 and j != 0:
                    local_shift += self.size[1] - self.kernel.shape[-1]
                buf_matrix[shift + j + local_shift, j].assign(1.0)
            buf_matrices.append(buf_matrix)
        tr_matr = []
        for k in range(self.in_channels):
            k_buf = []
            for t in range(self.out_channels):
                t_buf = []
                for i in range(out_size0 * out_size1):
                    buf = tf.matmul(buf_matrices[i], tf.reshape(self.kernel[k, t], [self.kernel.shape[2] * self.kernel.shape[3], 1]))
                    t_buf.append(buf)
                k_buf.append(t_buf)
            tr_matr.append(k_buf)
        tr_ker = tf.stack(tr_matr)
        return tf.transpose(tf.squeeze(tr_ker), perm=[0, 1, 3, 2])
    
    def call(self, inputs):
        '''
        Forward pass.
        Input:
            inputs - tf.tensor, input image. Should be 4D tensor of shape
            [batch_size, height, width, in_channels]
        Output:
            conv - tf.tensor, output image
        '''
        # make input shape torch-like
        self.inp = tf.transpose(inputs, perm=[0, 3, 1, 2])
        self.size = (self.inp.shape[2] + self.kernel.shape[2] - 1, self.inp.shape[3] + self.kernel.shape[3] - 1)
        transposed_kernel = self.transpose_kernel()
        # do smth
        res_matr = []
        for b in range(self.inp.shape[0]):
            b_matr = []
            for i in range(self.out_channels):
                for j in range(self.in_channels):
                    if j == 0:
                        buf = tf.matmul(transposed_kernel[j, i], tf.reshape(self.inp[b, j], [self.inp.shape[2] * self.inp.shape[3], 1]))
                    else:
                        buf += tf.matmul(transposed_kernel[j, i], tf.reshape(self.inp[b, j], [self.inp.shape[2] * self.inp.shape[3], 1]))
                b_matr.append(tf.squeeze(buf))
                # res[b, i] = tf.add(res[b, i], self.bias[i])
            res_matr.append(b_matr)
        res = tf.stack(res_matr)
        buf = tf.reshape(res, [self.inp.shape[0], self.out_channels, self.size[0], self.size[1]])
        return tf.transpose(buf, perm=[0, 2, 3, 1])
    
    def backward(self, grad_out):
        '''
        Backward pass.
        Input:
            grad_out - tf.tensor, gradient from next layer
        Output:
            grad_input - tf.tensor, grad_output multiplied by gradient of
            forward's output wrt to input
            grad_kernel - tf.tensor, grad_output multiplied by gradient of
            forward's output wrt to kernel
        '''
        inp = self.inp
        transposed_kernel = self.transpose_kernel()
        
        out_size0 = (self.size[0] // self.kernel.shape[2] - 1) * self.kernel.shape[2] + self.size[0] % self.kernel.shape[2] + 1
        out_size1 = (self.size[1] // self.kernel.shape[3] - 1) * self.kernel.shape[3] + self.size[1] % self.kernel.shape[3] + 1
        grad_input = grad_kernel = None
        
        grad_output = tf.transpose(grad_out, [0, 3, 1, 2])
        # calc grad wrt input
        grad_input = tf.zeros((grad_output.shape[0], self.in_channels,
                               out_size0, out_size1))
        grad_inp_matr = []
        for b in range(grad_output.shape[0]):
            b_matr = []
            for j in range(self.in_channels):
                for i in range(self.out_channels):
                    if i == 0:
                        buf = tf.matmul(tf.reshape(grad_output[b, i], [1, self.size[0] * self.size[1]]),
                                        transposed_kernel[j, i])
                    else:
                        buf += tf.matmul(tf.reshape(grad_output[b, i], [1, self.size[0] * self.size[1]]),
                                         transposed_kernel[j, i])
                b_matr.append(tf.squeeze(buf))
            grad_inp_matr.append(b_matr)
        gr_inp = tf.stack(grad_inp_matr)
        grad_input = tf.reshape(gr_inp,
                                [grad_output.shape[0], self.in_channels,
                                 out_size0, out_size1])
        # calc grad wrt kernel
        for b in range(grad_output.shape[0]):
            b_matr = []
            for i in range(self.out_channels):
                i_matr = []
                for j in range(self.in_channels):
                    buf = tf.matmul(tf.reshape(grad_output[b, i], [transposed_kernel.shape[-2], 1]),
                                    tf.reshape(inp[b, j], [1, transposed_kernel.shape[-1]]))
                    i_matr.append(buf)
                b_matr.append(i_matr)
        grad_kernel1 = tf.transpose(tf.stack(b_matr), [1, 0, 3, 2])
        gr_k_matr = []
        for j in range(grad_kernel1.shape[-2]):
            j_matr = []
            if j % out_size1 == 0:
                shift = (j // out_size1) * self.size[1]
            else:
                shift += 1
            for i in range(self.kernel.shape[-2]):
                buf = grad_kernel1[:, :, j, shift+i*self.size[1]: shift+i*self.size[1]+self.kernel.shape[-1]]
                j_matr.append(buf)
            gr_k_matr.append(j_matr)
        grad_kernel = tf.reduce_sum(tf.transpose(tf.stack(gr_k_matr), [0, 2, 1, 3, 4]), 0)
        return grad_input, grad_kernel


# testing parameters
BATCH_SIZE = 1
IN_CHANNELS = 2
OUT_CHANNELS = 3
SIZE_IN1 = 2
SIZE_IN2 = 2
KERNEL_SIZE1 = 3
KERNEL_SIZE2 = 3
SIZE_OUT1 = SIZE_IN1 + KERNEL_SIZE1 - 1
SIZE_OUT2 = SIZE_IN2 + KERNEL_SIZE2 - 1

def test_forward():
    '''
    Function for testing transposed convolution forward pass.
    '''
    inp = tf.random.normal((BATCH_SIZE, SIZE_IN1, SIZE_IN2, IN_CHANNELS), mean=0.0, stddev=1.0,
                       dtype=tf.dtypes.float32, seed=SEED, name=None)
    kernel = tf.random.normal((KERNEL_SIZE1, KERNEL_SIZE2, OUT_CHANNELS, IN_CHANNELS), mean=0.0, stddev=1.0,
                              dtype=tf.dtypes.float32, seed=SEED, name=None)
    out_shape = tf.constant([BATCH_SIZE, SIZE_OUT1, SIZE_OUT2, OUT_CHANNELS])
    res = tf.nn.conv2d_transpose(inp, kernel, out_shape, strides=[1, 1],
                                 padding='VALID')
    cust_layer = TransposedConvTF(IN_CHANNELS, OUT_CHANNELS, (KERNEL_SIZE1, KERNEL_SIZE2))
    cust_layer.build()
    cust_layer.kernel = tf.transpose(kernel, [3, 2, 0, 1])
    cust_res = cust_layer.call(inp)
    print('res', res)
    print('my_res', cust_res)
    # Check results on equality
    tf.debugging.assert_near(res, cust_res, rtol=1e-3, atol=1e-3)


def test_backward():
    '''
    Function for testing transposed convolution backward pass.
    '''
    inp = tf.random.normal((BATCH_SIZE, SIZE_IN1, SIZE_IN2, IN_CHANNELS), mean=0.0, stddev=1.0,
                       dtype=tf.dtypes.float32, seed=SEED, name=None)
    kernel = tf.random.normal((KERNEL_SIZE1, KERNEL_SIZE2, OUT_CHANNELS, IN_CHANNELS), mean=0.0, stddev=1.0,
                              dtype=tf.dtypes.float32, seed=SEED, name=None)
    out_shape = tf.constant([BATCH_SIZE, SIZE_OUT1, SIZE_OUT2, OUT_CHANNELS])
    res = tf.nn.conv2d_transpose(inp, kernel, out_shape, strides=[1, 1],
                                 padding='VALID')
    cust_layer = TransposedConvTF(IN_CHANNELS, OUT_CHANNELS, (KERNEL_SIZE1, KERNEL_SIZE2))
    cust_layer.build()
    cust_layer.kernel = tf.transpose(kernel, [3, 2, 0, 1])
    cust_res = cust_layer.call(inp)
    gr_out = tf.random.normal((BATCH_SIZE, SIZE_OUT1, SIZE_OUT2, OUT_CHANNELS), mean=0.0, stddev=1.0,
                          dtype=tf.dtypes.float32, seed=SEED, name=None)
    grad_input, grad_kernel = cust_layer.backward(gr_out)
    print('Shapes of gradient wrt input and input:', grad_input.shape, inp.shape)
    print('Shapes of gradient wrt kernel and kernel:', grad_kernel.shape, tf.transpose(kernel, [3, 2, 0, 1]).shape)


def main():
    test_forward()
    test_backward()


if __name__ == '__main__':
    main()
