from __future__ import absolute_import, print_function

import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tvm
import vta
import tvm
from tvm import te

from matplotlib import pyplot as plt
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data, utils
from tvm import rpc, autotvm, relay
from tvm.relay.testing import yolo_detection, darknet
from tvm.relay.testing.darknet import __darknetffi__
from tvm.contrib import graph_runtime, graph_runtime, util
from tvm.contrib.download import download_testdata
from vta.testing import simulator
from vta.top import graph_pack
from tvm.contrib.debugger import debug_runtime
import json
# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")


env = vta.get_env()
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu
if env.TARGET not in ["sim", "tsim"]:
    # Get remote from tracker node if environment variable is set.
    # To set up the tracker, you'll need to follow the "Auto-tuning
    # a convolutional network for VTA" tutorial.
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    # Otherwise if you have a device you want to program directly from
    # the host, make sure you've set the variables below to the IP of
    # your board.
    device_host = os.environ.get("VTA_PYNQ_RPC_HOST", "192.168.2.99")
    device_port = os.environ.get("VTA_PYNQ_RPC_PORT", "9091")
    if not tracker_host or not tracker_port:
        remote = rpc.connect(device_host, int(device_port))
    else:
        remote = autotvm.measure.request_remote(env.TARGET,
                                                tracker_host,
                                                int(tracker_port),
                                                timeout=10000)
    # Reconfigure the JIT runtime and FPGA.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
    reconfig_start = time.time()
    vta.reconfig_runtime(remote)
    vta.program_fpga(remote, bitstream=None)
    reconfig_time = time.time() - reconfig_start
    print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

# In simulation mode, host the RPC server locally.
else:
    remote = rpc.LocalSession()

# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

supported_model = [
    'ssd_512_resnet50_v1_voc',
    'faster_rcnn_resnet50_v1b_voc',
    'ssd_512_resnet50_v1_coco',
    'ssd_512_resnet101_v2_voc',
    'ssd_512_mobilenet1.0_voc',
    'ssd_512_mobilenet1.0_coco',
    'ssd_300_vgg16_atrous_voc'
    'ssd_512_vgg16_atrous_coco',
]

model_name = supported_model[1]
dshape = (1, 3, 512, 512)

######################################################################
# Download and pre-process demo image

im_fname = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                             'gluoncv/detection/street_small.jpg?raw=true',
                             'street_small.jpg', module='data')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

######################################################################
# Convert and compile model for CPU.

block = model_zoo.get_model(model_name, pretrained=True)

def build(target):
    mod, params = relay.frontend.from_mxnet(block, {"data": dshape})
    with relay.build_config(opt_level=3,disabled_pass={"AlterOpLayout"}):
        with relay.quantize.qconfig(global_scale=8.0,
                                    skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)
        # Perform graph packing and constant folding for VTA target
        assert env.BLOCK_IN == env.BLOCK_OUT
        mod = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name='nn.max_pool2d',
            stop_name='cast',
            start_name_idx = 3,
            stop_name_idx = 388)
        with vta.build_config():
            graph, lib, params = relay.build(
                mod,
                target=target,
                params=params,
                target_host=env.target_host)
    return graph, lib, params

######################################################################
# Create TVM runtime and do inference

def run(graph, lib, params, ctx):
    # Build TVM runtime
    temp = util.tempdir()
    lib.save(temp.relpath("graphlib.o"))
    file_name = "deploy.so"
    path_lib = temp.relpath(file_name)
    lib.export_library(path_lib)
    remote.upload(temp.relpath("graphlib.o"))
    lib = remote.load_module("graphlib.o")
    m = graph_runtime.create(graph, lib, ctx)
    tvm_input = tvm.nd.array(x.asnumpy(), ctx=ctx)
    m.set_input('data', tvm_input)
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    return class_IDs, scores, bounding_boxs


graph, lib, params = build(target)
class_IDs, scores, bounding_boxs = run(graph, lib, params, ctx)

######################################################################
# Display result

ax = utils.viz.plot_bbox(img, bounding_boxs.asnumpy()[0], scores.asnumpy()[0],
                         class_IDs.asnumpy()[0], class_names=block.classes)
plt.show()

