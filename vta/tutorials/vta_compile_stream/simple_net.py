import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing
import argparse, json, os, requests, sys, time
from io import BytesIO
from os.path import join, isfile
from PIL import Image

from mxnet.gluon.model_zoo import vision
import numpy as np
from matplotlib import pyplot as plt

import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download
from tvm.contrib.debugger import debug_runtime
from tvm.relay import transform

import vta
from vta.testing import simulator
from vta.top import graph_pack

env = vta.get_env()

# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu
remote = rpc.LocalSession()
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

out_channels = 16
batch_size = 1
pixel = 1

data = relay.var("data", relay.TensorType((batch_size, 16, pixel, pixel), "float32"))
weight = relay.var("weight")
bn_gamma = relay.var("bn_gamma")
bn_beta = relay.var("bn_beta")
bn_mmean = relay.var("bn_mean")
bn_mvar = relay.var("bn_var")

simple_net = relay.nn.conv2d(data=data, weight=weight, kernel_size=(3,3), channels=out_channels, padding=(1, 1))
simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

data_shape = (batch_size, 16, pixel, pixel)
net, params = testing.create_workload(simple_net)
print("***************workload created***************")
print(net)  

with tvm.transform.PassContext(opt_level=3):
    with relay.quantize.qconfig(global_scale=8.0,
                                skip_conv_layers=[]):
        net = relay.quantize.quantize(net, params=params)
    print(net)
    # Perform graph packing and constant folding for VTA target
    assert env.BLOCK_IN == env.BLOCK_OUT
    net = graph_pack(
                net["main"],
                env.BATCH,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name="multiply",
                stop_name="cast",
                start_name_idx=0,
                stop_name_idx=10)
    print(net)
print("***************build started***************")
with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}, debug_flag=6):
    graph, lib, params = relay.build(
        net, target=target,
        params=params, target_host=env.target_host)
print("***************build finished***************")
print(lib)
temp = util.tempdir()
lib.save(temp.relpath("graphlib.o"))
remote.upload(temp.relpath("graphlib.o"))
lib = remote.load_module("graphlib.o")

# Graph runtime
m = graph_runtime.create(graph, lib, ctx)
image = np.random.rand(1, 16, pixel, pixel)
# Set the network parameters and inputs
m.set_input(**params)
m.set_input('data', image)
m.run()
