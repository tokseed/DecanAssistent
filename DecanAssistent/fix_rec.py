import sys
import onnx
from onnx.tools import update_model_dims

src, dst, width = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = onnx.load(src)
inp = m.graph.input[0].name
out = m.graph.output[0].name
classes = m.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
del m.graph.value_info[:]
m = update_model_dims.update_inputs_outputs_dims(m, {inp: [1, 3, 48, width]}, {out: [1, "T", classes]})
m = onnx.shape_inference.infer_shapes(m)
try:
    import onnxsim
    m, ok = onnxsim.simplify(m)
    print("onnxsim:", "ok" if ok else "check failed")
except ImportError:
    print("onnxsim: not installed")
onnx.save(m, dst)
dims = [d.dim_value or d.dim_param for d in m.graph.output[0].type.tensor_type.shape.dim]
print("output dims:", dims)
print("ir_version:", m.ir_version, "opset:", [(o.domain, o.version) for o in m.opset_import])
print("ops:", sorted({n.op_type for n in m.graph.node}))
