import argparse
import json
import sys
import os
import time
import logging
from tensorflow.python.platform import gfile
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
from six import iteritems

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ENGINE_FILE = "charrnn.engine"


INPUT_BLOB_NAME = "inputData"
HIDDEN_IN_BLOB_NAME = "hiddenIn"
HIDDEN_OUT_BLOB_NAME = "hiddenOut"
CELL_IN_BLOB_NAME = "cellIn"
CELL_OUT_BLOB_NAME = "cellOut"
OUTPUT_BLOB_NAME = "softmax"

INPUT_SHAPE = (batch_size, 1, vocab_size)
HIDDEN_SHAPE = (batch_size, num_layers, hidden_size)
CELL_SHAPE = (batch_size, num_layers, hidden_size)
OUTPUT_SHAPE = (batch_size, 1, vocab_size)

gSizes = {INPUT_BLOB_NAME: INPUT_SHAPE,
HIDDEN_IN_BLOB_NAME: HIDDEN_SHAPE,
HIDDEN_OUT_BLOB_NAME: HIDDEN_SHAPE,
CELL_IN_BLOB_NAME: CELL_SHAPE,
CELL_OUT_BLOB_NAME: CELL_SHAPE,
OUTPUT_BLOB_NAME: OUTPUT_SHAPE}


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, name):
        self.host = host_mem
        self.device = device_mem
        self.name = name

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device) + "\nName:\n" + str(self.name)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    for binding in engine:
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # allocate host and device buffers
        host_mem = cuda.pagelocked_empty(gSizes[binding], dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # append the device buffer to device bindings
        bindings.append(int(device_mem))
        # append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, binding))
    return inputs, outputs, bindings, stream                

def do_inference(context, bindings, inputs, outputs, stream, batch_size):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    re = context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]
    
    
    
