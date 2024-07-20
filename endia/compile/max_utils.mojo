from endia import Array, Node, ArrayShape, ShapeNode
from endia.functional._utils import contiguous, array_shape_to_list
from compile import *


from max.engine import InferenceSession, Model, TensorMap, EngineNumpyView
from max.graph import Graph, TensorType, ops, Symbol, Dim, Type
from max.tensor import Tensor, TensorShape, TensorSpec
from python import Python
from .python_utils import array_to_numpy, tensor_to_array


fn top_order(inout curr: Array) -> List[Array]:
    var trace = List[Array]()
    reset_node_id_recursive(curr)
    top_order_rec(curr, trace)
    return trace


fn to_tensor(arg: Array) raises -> Tensor[DType.float32]:
    var shape = TensorShape(List(arg.shape()))
    var tensor = Tensor[DType.float32](shape)
    for i in range(arg.size()):
        tensor.store(i, arg.load(i))
    return tensor


fn make_equal_rank(
    arg: Symbol, arg_shape: List[Int], comp_shape: List[Int]
) raises -> Symbol:
    var diff = len(comp_shape) - len(arg_shape)
    if diff > 0:
        var res = arg
        for _ in range(diff):
            res = ops.unsqueeze(res, 0)
        return res
    return arg


def build_graph(
    args: List[Array], outputs: List[Array], trace: List[Array]
) -> Graph:
    var arg_specs = List[Type]()
    for arg in args:
        arg_specs.append(TensorType(TensorSpec(DType.float32, arg[].shape())))
    var out_specs = List[Type]()
    for out in outputs:
        out_specs.append(TensorType(TensorSpec(DType.float32, out[].shape())))
    graph = Graph(name="subgraph", in_types=arg_specs, out_types=out_specs)

    var symbol_trace = List[Symbol]()

    var args_idx = Dict[String, Int]()
    for i in range(len(args)):
        args_idx[str(args[i].id())] = i

    var output_symbols = List[Symbol]()

    for array in trace:
        var tmp_args = List[Array]()
        for arg in array[].args():
            tmp_args.append(
                arg[]  # if (not arg[].name() == "brdcst" or len(arg[].args()) == 0) else arg[].args()[0]
            )

        if len(tmp_args) == 0:
            var idx_in_args = args_idx[str(array[].id())]
            symbol_trace.append(graph[idx_in_args])
            continue

        elif array[].is_view():
            var arg0 = symbol_trace[
                tmp_args[0].id()
            ]  # if tmp_args[0].has_fxgraph() else graph.constant(to_tensor(tmp_args[0]))
            if array[].name() == "brdcst":
                # symbol_trace.append(symbol_trace[tmp_args[0].id()])
                var zero_const = graph.constant(
                    Tensor[DType.float32](array[].shape(), 0)
                )
                symbol_trace.append(ops.add(arg0, zero_const))
            elif array[].name() == "permute":
                symbol_trace.append(ops.transpose(arg0, -1, -2))
            elif array[].name() == "squeeze":
                var all_axis = array_shape_to_list(
                    array[].array_shape().args()[1]
                )
                for i in range(len(all_axis)):
                    arg0 = ops.squeeze(arg0, all_axis[i] - i)
                symbol_trace.append(arg0)
            elif array[].name() == "unsqueeze":
                var all_axis = array_shape_to_list(
                    array[].array_shape().args()[1]
                )
                for axis in all_axis:
                    arg0 = ops.unsqueeze(arg0, axis[])
                symbol_trace.append(arg0)
            elif array[].name() == "permute":
                symbol_trace.append(ops.transpose(arg0, -1, -2))
            else:
                print("Unknown view op:", array[].name())
            continue

        elif array[].name() == "reduce_add":
            var arg0 = symbol_trace[tmp_args[0].id()]
            var in_shape = tmp_args[0].shape()
            var all_axis = array_shape_to_list(array[].array_shape().args()[1])
            for i in range(len(all_axis)):
                var axis = all_axis[i]
                # MAX currently only has a mean op and no general reduce_add op, hence we need to multiply by the divisor to emulate reduce_add
                var divisor = in_shape[axis]
                var divisor_constant_value = Tensor[DType.float32](
                    TensorShape(1), divisor
                )
                var divisor_constant = graph.constant(divisor_constant_value)
                arg0 = ops.mean(arg0, axis) * divisor_constant
            symbol_trace.append(arg0)
            continue

        elif len(tmp_args) == 1:
            # unary op
            arg0 = symbol_trace[tmp_args[0].id()]
            if array[].name() == "abs":
                symbol_trace.append(ops.abs(arg0))
            # elif array[].name() == "acos":
            #     symbol_trace.append(ops.acos(arg0))
            # elif array[].name() == "asin":
            #     symbol_trace.append(ops.asin(arg0))
            # elif array[].name() == "atan":
            #     symbol_trace.append(ops.atan(arg0))
            elif array[].name() == "cos":
                symbol_trace.append(ops.cos(arg0))
            # elif array[].name() == "cosh":
            #     symbol_trace.append(ops.cosh(arg0))
            elif array[].name() == "exp":
                symbol_trace.append(ops.exp(arg0))
            elif array[].name() == "log":
                symbol_trace.append(ops.log(arg0))
            elif array[].name() == "neg":
                symbol_trace.append(-arg0)
            elif array[].name() == "reciprocal":
                symbol_trace.append(1 / arg0)
            elif array[].name() == "relu":
                symbol_trace.append(ops.relu(arg0))
            elif array[].name() == "sigmoid":
                symbol_trace.append(ops.sigmoid(arg0))
            # elif array[].name() == "sign":
            #     symbol_trace.append(ops.sign(arg0))
            elif array[].name() == "sin":
                symbol_trace.append(ops.sin(arg0))
            # elif array[].name() == "sinh":
            #     symbol_trace.append(ops.sinh(arg0))
            elif array[].name() == "sqrt":
                symbol_trace.append(ops.sqrt(arg0))
            # elif array[].name() == "tan":
            #     symbol_trace.append(ops.tan(arg0))
            elif array[].name() == "tanh":
                symbol_trace.append(ops.tanh(arg0))

            else:
                print("Unknown unary op:", array[].name())

        elif len(tmp_args) == 2:
            var arg1 = symbol_trace[tmp_args[0].id()]
            var arg2 = symbol_trace[tmp_args[1].id()]

            # binary ops
            if array[].name() == "add":
                symbol_trace.append(ops.add(arg1, arg2))
            elif array[].name() == "sub":
                symbol_trace.append(ops.sub(arg1, arg2))
            elif array[].name() == "mul":
                symbol_trace.append(ops.mul(arg1, arg2))
            elif array[].name() == "div":
                symbol_trace.append(ops.div(arg1, arg2))
            elif array[].name() == "pow_to":
                symbol_trace.append(ops.pow(arg1, arg2))
            elif array[].name() == "matmul":
                symbol_trace.append(ops.matmul(arg1, arg2))

            # comparison ops
            elif array[].name() == "greater_equal":
                symbol_trace.append(ops.greater_equal(arg1, arg2))
            elif array[].name() == "greater":
                symbol_trace.append(ops.greater(arg1, arg2))
            elif array[].name() == "equal":
                symbol_trace.append(ops.equal(arg1, arg2))
            elif array[].name() == "not_equal":
                symbol_trace.append(ops.not_equal(arg1, arg2))
            elif array[].name() == "less":
                symbol_trace.append(ops.greater(arg2, arg1))
            elif array[].name() == "less_equal":
                symbol_trace.append(ops.greater_equal(arg2, arg1))
            else:
                print("Unknown binary op:", array[].name())
        else:
            raise "Unknown op:" + array[].name()

    for output in outputs:
        output_symbols.append(symbol_trace[output[].id()])

    graph.output(output_symbols)
    return graph


fn build_model(
    args: List[Array], outputs: List[Array], trace: List[Array]
) raises -> Model:
    print("JIT compiling a new subgraph...")
    var graph = build_graph(args, outputs, trace)
    var session = InferenceSession()
    var model = session.load(graph)
    return model


def execute_model(
    args: List[Array], outputs: List[Array], model: Model
) -> List[Array]:
    # Convert args to numpy arrays and store in numpy dict with key "input0", "input1", ...
    var np = Python.import_module("numpy")
    var numpy_dict = Python.dict()
    for id in range(len(args)):
        var arg = args[id]
        var np_array = array_to_numpy(arg, np)
        numpy_dict["input" + str(id)] = np_array

    # Execute_max_graph the model
    var results = model.execute(numpy_dict^)

    # Put all intermediate nodes into the output list
    var array_outputs = List[Array]()
    for i in range(len(outputs)):
        var output = results.get[DType.float32]("output" + str(i))
        array_outputs.append(tensor_to_array(output))
    return array_outputs
