#!/usr/bin/env python3
"""Generates the sample ONNX models (random weights) shipped with TensorView VR.

Usage: python tools/make_model.py <output_dir>      (writes 3 models)
       python tools/make_model.py <file.onnx>       (writes only the ResNet-style model)
Requires: pip install onnx numpy
"""
import os
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

rng = np.random.default_rng(7)


class B:
    """Tiny graph builder."""

    def __init__(self):
        self.nodes, self.inits = [], []

    def W(self, name, *shape, scale=0.05):
        self.inits.append(numpy_helper.from_array((rng.standard_normal(shape) * scale).astype(np.float32), name))
        return name

    def op(self, typ, ins, name, out=None, **attrs):
        out = out or name + "_out"
        self.nodes.append(helper.make_node(typ, ins, [out], name=name, **attrs))
        return out

    def conv(self, x, name, cin, cout, k, s, p, group=1):
        return self.op("Conv", [x, self.W(name + "_w", cout, cin // group, k, k), self.W(name + "_b", cout)], name,
                       kernel_shape=[k, k], strides=[s, s], pads=[p] * 4, group=group)

    def relu(self, x, name):
        return self.op("Relu", [x], name)

    def save(self, path, gname, inp, outp, out_name):
        g = helper.make_graph(self.nodes, gname,
                              [helper.make_tensor_value_info(inp[0], TensorProto.FLOAT, inp[1])],
                              [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, outp)], self.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 8
        m = shape_inference.infer_shapes(m)
        onnx.checker.check_model(m)
        onnx.save(m, path)
        print("saved", path, "nodes:", len(self.nodes))


def resnet(path):
    b = B()

    def block(x, name, cin, cout, stride):
        a = b.relu(b.conv(x, name + "_conv1", cin, cout, 3, stride, 1), name + "_relu1")
        c = b.conv(a, name + "_conv2", cout, cout, 3, 1, 1)
        skip = x if (stride == 1 and cin == cout) else b.conv(x, name + "_skip", cin, cout, 1, stride, 0)
        s = b.op("Add", [c, skip], name + "_add")
        return b.relu(s, name + "_relu2")

    y = b.relu(b.conv("input", "stem_conv", 3, 32, 7, 2, 3), "stem_relu")
    y = b.op("MaxPool", [y], "stem_pool", kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    for name, cin, cout, st in [("layer1_block1", 32, 32, 1), ("layer1_block2", 32, 32, 1),
                                ("layer2_block1", 32, 64, 2), ("layer2_block2", 64, 64, 1),
                                ("layer3_block1", 64, 128, 2), ("layer3_block2", 128, 128, 1)]:
        y = block(y, name, cin, cout, st)
    y = b.op("GlobalAveragePool", [y], "global_pool")
    y = b.op("Flatten", [y], "flatten", axis=1)
    y = b.op("Gemm", [y, b.W("fc_w", 1000, 128), b.W("fc_b", 1000)], "classifier_fc", transB=1)
    y = b.op("Softmax", [y], "softmax", out="probs", axis=1)
    b.save(path, "vision_resnet_pro", ("input", [1, 3, 224, 224]), [1, 1000], "probs")


def mobilenet(path):
    b = B()

    def dsblock(x, name, cin, cout, stride):  # depthwise 3x3 + pointwise 1x1
        d = b.relu(b.conv(x, name + "_dw", cin, cin, 3, stride, 1, group=cin), name + "_dw_relu")
        return b.relu(b.conv(d, name + "_pw", cin, cout, 1, 1, 0), name + "_pw_relu")

    def inverted(x, name, c, exp):  # inverted residual with skip
        e = b.relu(b.conv(x, name + "_expand", c, c * exp, 1, 1, 0), name + "_expand_relu")
        d = b.relu(b.conv(e, name + "_dw", c * exp, c * exp, 3, 1, 1, group=c * exp), name + "_dw_relu")
        p = b.conv(d, name + "_project", c * exp, c, 1, 1, 0)
        return b.op("Add", [x, p], name + "_add")

    y = b.relu(b.conv("input", "stem_conv", 3, 16, 3, 2, 1), "stem_relu")
    y = dsblock(y, "ds1", 16, 24, 2)
    y = inverted(y, "ir1", 24, 3)
    y = dsblock(y, "ds2", 24, 48, 2)
    y = inverted(y, "ir2", 48, 3)
    y = dsblock(y, "ds3", 48, 96, 2)
    y = inverted(y, "ir3", 96, 2)
    y = b.op("GlobalAveragePool", [y], "global_pool")
    y = b.op("Flatten", [y], "flatten", axis=1)
    y = b.op("Gemm", [y, b.W("fc_w", 1000, 96), b.W("fc_b", 1000)], "classifier_fc", transB=1)
    y = b.op("Softmax", [y], "softmax", out="probs", axis=1)
    b.save(path, "mobilenet_lite", ("input", [1, 3, 224, 224]), [1, 1000], "probs")


def transformer(path):
    b = B()
    d, T = 64, 16

    def attention(x, name):
        q = b.op("MatMul", [x, b.W(name + "_wq", d, d)], name + "_q")
        k = b.op("MatMul", [x, b.W(name + "_wk", d, d)], name + "_k")
        v = b.op("MatMul", [x, b.W(name + "_wv", d, d)], name + "_v")
        kt = b.op("Transpose", [k], name + "_kT", perm=[0, 2, 1])
        s = b.op("MatMul", [q, kt], name + "_scores")
        a = b.op("Softmax", [s], name + "_softmax", axis=-1)
        c = b.op("MatMul", [a, v], name + "_context")
        o = b.op("MatMul", [c, b.W(name + "_wo", d, d)], name + "_proj")
        return b.op("Add", [x, o], name + "_residual")

    def ffn(x, name):
        h = b.relu(b.op("MatMul", [x, b.W(name + "_w1", d, d * 2)], name + "_up"), name + "_act")
        o = b.op("MatMul", [h, b.W(name + "_w2", d * 2, d)], name + "_down")
        return b.op("Add", [x, o], name + "_residual")

    y = b.op("MatMul", ["input", b.W("embed_w", 32, d)], "embed")
    for i in (1, 2):
        y = ffn(attention(y, f"block{i}_attn"), f"block{i}_ffn")
    y = b.op("ReduceMean", [y], "pool", axes=[1], keepdims=0)
    y = b.op("Gemm", [y, b.W("head_w", 10, d), b.W("head_b", 10)], "head", transB=1)
    y = b.op("Softmax", [y], "softmax", out="probs", axis=1)
    b.save(path, "tiny_transformer", ("input", [1, T, 32]), [1, 10], "probs")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    if target.endswith(".onnx"):
        resnet(target)
    else:
        os.makedirs(target, exist_ok=True)
        resnet(os.path.join(target, "vision_resnet_pro.onnx"))
        mobilenet(os.path.join(target, "mobilenet_lite.onnx"))
        transformer(os.path.join(target, "tiny_transformer.onnx"))
