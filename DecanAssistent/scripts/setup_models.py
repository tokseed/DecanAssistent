#!/usr/bin/env python3
"""Provision small deterministic ONNX models for DecanAssistent tests.

The project-specific production models are not part of the repository.  This
script therefore creates four tiny, valid ONNX graphs that are sufficient to
exercise model loading, NeoML inference, CTC decoding and detector post-
processing.  The graph is input-connected but has zero weights and a
deterministic bias, so the models are deliberately not intended to provide
real OCR or visual recognition quality.

If the optional ``onnx`` Python package is installed, its helper/checker API is
used.  A small protobuf encoder is kept as a dependency-free fallback so a
fresh CMake build does not require torch, onnxruntime or network access.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Sequence


PRINTED_ALPHABET = (
    " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "0123456789.,:;!?-№()/%«»\""
)
HANDWRITING_ALPHABET = "0123456789./№-"


def _varint(value: int) -> bytes:
    """Encode an unsigned protobuf varint."""
    if value < 0:
        value &= (1 << 64) - 1
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field_varint(number: int, value: int) -> bytes:
    return _varint((number << 3) | 0) + _varint(value)


def _field_bytes(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _field_string(number: int, value: str) -> bytes:
    return _field_bytes(number, value.encode("utf-8"))


def _field_message(number: int, value: bytes) -> bytes:
    return _field_bytes(number, value)


def _tensor_proto(name: str, shape: Sequence[int], values: Sequence[float],
                  data_type: int = 1) -> bytes:
    # TensorProto: dims=1, data_type=2, name=8, raw_data=9.
    # FLOAT is 1 and INT64 is 7 in the ONNX TensorProto enum.
    result = bytearray()
    for dimension in shape:
        result += _field_varint(1, int(dimension))
    result += _field_varint(2, data_type)
    result += _field_string(8, name)
    if data_type == 7:
        raw = struct.pack("<" + "q" * len(values), *(int(value) for value in values))
    else:
        raw = struct.pack("<" + "f" * len(values), *values)
    result += _field_bytes(9, raw)
    return bytes(result)


def _tensor_shape(shape: Sequence[int]) -> bytes:
    result = bytearray()
    for dimension in shape:
        # TensorShapeProto.Dimension: dim_value=1.
        dimension_message = _field_varint(1, int(dimension))
        result += _field_message(1, dimension_message)
    return bytes(result)


def _tensor_type(shape: Sequence[int]) -> bytes:
    # TypeProto.tensor_type=1; Tensor.elem_type=1 (FLOAT), shape=2.
    tensor = _field_varint(1, 1) + _field_message(2, _tensor_shape(shape))
    return _field_message(1, tensor)


def _value_info(name: str, shape: Sequence[int]) -> bytes:
    # ValueInfoProto: name=1, type=2.
    return _field_string(1, name) + _field_message(2, _tensor_type(shape))


def _attribute_int(name: str, value: int) -> bytes:
    # AttributeProto: name=1, i=3, type=20 (INT).
    return (_field_string(1, name) + _field_varint(3, value) +
            _field_varint(20, 2))


def _node(inputs: Sequence[str], output: str, op_type: str,
          attributes: Sequence[bytes] = ()) -> bytes:
    # NodeProto: input=1, output=2, name=3, op_type=4, attribute=5.
    result = bytearray()
    for input_name in inputs:
        result += _field_string(1, input_name)
    result += _field_string(2, output)
    result += _field_string(3, "decan_dummy_" + op_type.lower())
    result += _field_string(4, op_type)
    for attribute in attributes:
        result += _field_message(5, attribute)
    return bytes(result)


def _network_graph(input_name: str, input_shape: Sequence[int], output_name: str,
                   output_shape: Sequence[int], output_values: Sequence[float]) -> bytes:
    """Create a small input-connected graph with deterministic output values.

    A standalone ONNX Constant is legal, but older NeoML importers can crash
    while executing a graph whose declared input is disconnected from the
    output.  The dummy graph therefore uses GlobalAveragePool -> Flatten ->
    Gemm.  The Gemm weights are zero and its bias is the requested output, so
    the graph is connected to the image while remaining deterministic.
    """
    channels = int(input_shape[1]) if len(input_shape) >= 2 else 1
    output_size = 1
    for dimension in output_shape:
        output_size *= int(dimension)
    if output_size != len(output_values):
        raise ValueError("dummy output shape does not match output values")

    nodes = [
        _node([input_name], "pooled", "GlobalAveragePool"),
        _node(["pooled"], "flattened", "Flatten", [_attribute_int("axis", 1)]),
    ]
    initializers = [
        _tensor_proto("weights", [channels, output_size],
                      [0.0] * (channels * output_size)),
        _tensor_proto("bias", [output_size], output_values),
    ]
    if len(output_shape) == 2:
        nodes.append(_node(["flattened", "weights", "bias"], output_name, "Gemm"))
    else:
        nodes.append(_node(["flattened", "weights", "bias"], "linear", "Gemm"))
        initializers.append(_tensor_proto("output_shape", [len(output_shape)],
                                          output_shape, data_type=7))
        nodes.append(_node(["linear", "output_shape"], output_name, "Reshape"))

    # GraphProto: node=1, name=2, initializer=5, input=11, output=12.
    result = bytearray()
    for node in nodes:
        result += _field_message(1, node)
    result += _field_string(2, "decan_dummy_graph")
    for initializer in initializers:
        result += _field_message(5, initializer)
    result += _field_message(11, _value_info(input_name, input_shape))
    result += _field_message(12, _value_info(output_name, output_shape))
    return bytes(result)


def _model(input_name: str, input_shape: Sequence[int], output_name: str,
           output_shape: Sequence[int], output_values: Sequence[float]) -> bytes:
    # ModelProto: ir_version=1, producer_name=2, graph=7, opset_import=8.
    # IR 7 and opset 11 are accepted by the NeoML version vendored in this repo.
    opset = _field_varint(2, 11)  # default domain is the empty string
    graph = _network_graph(input_name, input_shape, output_name, output_shape, output_values)
    return (_field_varint(1, 7) +
            _field_string(2, "DecanAssistent dummy model") +
            _field_message(7, graph) +
            _field_message(8, opset))


def _ctc_logits(text: str, alphabet: str) -> tuple[list[float], list[int]]:
    """Return flattened [time, classes] logits and the output shape."""
    symbols = list(alphabet)
    class_count = len(symbols) + 1  # class zero is the CTC blank
    time_classes: list[int] = []
    for character in text:
        if character not in symbols:
            raise ValueError(f"Character {character!r} is not present in the dummy alphabet")
        # Separate equal neighbouring symbols with a blank timestep.
        if time_classes:
            time_classes.append(0)
        time_classes.append(symbols.index(character) + 1)
    time = max(1, len(time_classes))
    logits = []
    for selected in time_classes or [0]:
        logits.extend(10.0 if index == selected else -10.0
                      for index in range(class_count))
    return logits, [1, time, class_count]


def _onnx_model(input_name: str, input_shape: Sequence[int], output_name: str,
                output_shape: Sequence[int], values: Sequence[float]):
    """Build an input-connected ONNX ModelProto when onnx is available."""
    import onnx  # type: ignore
    from onnx import TensorProto, helper  # type: ignore

    input_info = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, list(input_shape))
    output_info = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, list(output_shape))
    channels = int(input_shape[1]) if len(input_shape) >= 2 else 1
    output_size = 1
    for dimension in output_shape:
        output_size *= int(dimension)
    weights = helper.make_tensor(
        "weights", TensorProto.FLOAT, [channels, output_size],
        [0.0] * (channels * output_size))
    bias = helper.make_tensor(
        "bias", TensorProto.FLOAT, [output_size], list(values))
    nodes = [
        helper.make_node("GlobalAveragePool", [input_name], ["pooled"],
                         name="decan_dummy_global_average_pool"),
        helper.make_node("Flatten", ["pooled"], ["flattened"],
                         name="decan_dummy_flatten", axis=1),
    ]
    initializers = [weights, bias]
    if len(output_shape) == 2:
        nodes.append(helper.make_node(
            "Gemm", ["flattened", "weights", "bias"], [output_name],
            name="decan_dummy_gemm"))
    else:
        nodes.append(helper.make_node(
            "Gemm", ["flattened", "weights", "bias"], ["linear"],
            name="decan_dummy_gemm"))
        output_shape_tensor = helper.make_tensor(
            "output_shape", TensorProto.INT64, [len(output_shape)], list(output_shape))
        initializers.append(output_shape_tensor)
        nodes.append(helper.make_node(
            "Reshape", ["linear", "output_shape"], [output_name],
            name="decan_dummy_reshape"))
    graph = helper.make_graph(
        nodes, "decan_dummy_graph", [input_info], [output_info], initializers)
    model = helper.make_model(
        graph, producer_name="DecanAssistent dummy model",
        opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _write_model(path: Path, input_shape: Sequence[int], output_shape: Sequence[int],
                 values: Sequence[float], force: bool) -> str:
    if path.exists() and not force:
        return "kept"
    path.parent.mkdir(parents=True, exist_ok=True)
    input_name = "input"
    output_name = "output"
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        try:
            model = _onnx_model(input_name, input_shape, output_name, output_shape, values)
            # ModelProto has no portable ``save`` method.  Use the package
            # serializer explicitly so this path also works when ``onnx`` is
            # installed instead of the built-in protobuf fallback.
            import onnx  # type: ignore
            onnx.save(model, str(temporary))
            backend = "onnx"
        except ImportError:
            temporary.write_bytes(_model(input_name, input_shape, output_name,
                                         output_shape, values))
            backend = "builtin-protobuf"
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return backend


def provision(output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    printed_logits, printed_shape = _ctc_logits("ТЕСТ", PRINTED_ALPHABET)
    handwriting_logits, handwriting_shape = _ctc_logits("14.02.2025", HANDWRITING_ALPHABET)

    models = {
        "ocr_printed.onnx": ([1, 1, 32, 128], printed_shape, printed_logits),
        "ocr_handwriting.onnx": ([1, 1, 32, 128], handwriting_shape, handwriting_logits),
        # Detector output is [N, 6] with normalized x1,y1,x2,y2,score,class_id.
        "seal_stamp_detector.onnx": (
            [1, 3, 640, 640], [1, 6], [0.35, 0.05, 0.48, 0.16, 0.99, 0.0]),
        "coat_of_arms_detector.onnx": (
            [1, 3, 640, 640], [1, 6], [0.43, 0.05, 0.57, 0.18, 0.98, 0.0]),
    }
    for filename, (input_shape, output_shape, values) in models.items():
        result = _write_model(output_dir / filename, input_shape, output_shape, values, force)
        print(f"{result:>16} {output_dir / filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory in which the four ONNX files are created")
    parser.add_argument("--force", action="store_true",
                        help="replace existing files; disabled by default to protect real models")
    args = parser.parse_args()
    provision(args.output_dir, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
