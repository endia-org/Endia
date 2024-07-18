from endia import Array
from endia.utils import *
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math


###############################################################################################################
#                                            Array utils and setup
###############################################################################################################


@always_inline
fn compute_stride(shape: List[Int]) -> List[Int]:
    var stride = List[Int]()
    for _ in range(len(shape)):
        stride.append(1)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return stride


@always_inline
fn compute_storage_offset(
    indices: List[Int], stride: List[Int], storage_offset: Int
) -> Int:
    # if len(indices) != len(stride):
    #     raise "Indices and stride must have the same length"
    var result = storage_offset
    for i in range(len(indices)):
        result += indices[i] * stride[i]
    return result


@always_inline
fn compute_nd_index(index: Int, shape: List[Int]) -> List[Int]:
    # if len(shape) == 0:
    #     raise "Shape must have at least one dimension"
    var indices = List[Int]()
    var flat_idx = index
    for i in range(len(shape) - 1, -1, -1):
        indices.append(flat_idx % shape[i])
        flat_idx //= shape[i]
    indices.reverse()
    return indices


fn compute_indeces_for_matmul(
    i: Int, res: Array, lhs_b: Array, rhs_b: Array
) raises -> List[Int]:
    var nd_idx_res = compute_nd_index(i, res.shape())
    var nd_idx_lhs = List[Int]()
    var nd_idx_rhs = List[Int]()
    for j in range(len(nd_idx_res) - 2):
        nd_idx_lhs.append(nd_idx_res[j] if lhs_b.stride()[j] != 0 else 0)
        nd_idx_rhs.append(nd_idx_res[j] if rhs_b.stride()[j] != 0 else 0)
    nd_idx_lhs.append(nd_idx_res[len(nd_idx_res) - 2])
    nd_idx_lhs.append(0)
    nd_idx_rhs.append(0)
    nd_idx_rhs.append(nd_idx_res[len(nd_idx_res) - 1])
    var lhs_idx_tmp = compute_storage_offset(nd_idx_lhs, lhs_b.stride(), 0)
    var rhs_idx_tmp = compute_storage_offset(nd_idx_rhs, rhs_b.stride(), 0)
    return List(lhs_idx_tmp, rhs_idx_tmp)


fn execute_copy_raw(
    source_data: DTypePointer[DType.float32],
    dest_data: DTypePointer[DType.float32],
    val_shape: ArrayShape,
    is_complex: Bool,
) raises:
    var rank = val_shape.ndim()
    var shape = List[Int]()
    var stride = List[Int]()
    if rank == 1:
        shape.append(1)
        stride.append(0)
        shape.append(val_shape.shape()[0])
        stride.append(val_shape.stride()[0])
    else:
        shape = val_shape.shape()
        stride = val_shape.stride()
    rank = len(shape)
    var rows = shape[rank - 2]
    var cols = shape[rank - 1]
    var size = 1
    for i in range(rank):
        size *= shape[i]
    var flat_idx = 0
    for k in range(0, size, rows * cols):
        var nd_idx = compute_nd_index(k, shape)
        var base_idx = compute_storage_offset(
            nd_idx, stride, val_shape.storage_offset()
        )

        for i in range(rows):
            var i_idx = base_idx + i * stride[rank - 2]

            if is_complex:
                if stride[rank - 1] == 1:

                    @parameter
                    fn copy_v[simd_width: Int](j: Int):
                        var j_idx = i_idx + j * stride[rank - 1]
                        dest_data.store[width = 2 * simd_width](
                            flat_idx * 2,
                            source_data.load[width = 2 * simd_width](j_idx * 2),
                        )
                        flat_idx += simd_width

                    vectorize[copy_v, nelts[dtype]()](cols)
                else:
                    for j in range(cols):
                        var j_idx = i_idx + j * stride[rank - 1]
                        dest_data.store(
                            2 * flat_idx, source_data.load(2 * j_idx)
                        )
                        dest_data.store(
                            2 * flat_idx + 1, source_data.load(2 * j_idx + 1)
                        )
                        flat_idx += 1

            else:
                if stride[rank - 1] == 1:

                    @parameter
                    fn copy_v_complex[simd_width: Int](j: Int):
                        var j_idx = i_idx + j * stride[rank - 1]
                        dest_data.store[width=simd_width](
                            flat_idx, source_data.load[width=simd_width](j_idx)
                        )
                        flat_idx += simd_width

                    vectorize[copy_v_complex, nelts[dtype]()](cols)
                else:
                    for j in range(cols):
                        var j_idx = i_idx + j * stride[rank - 1]
                        dest_data.store(flat_idx, source_data.load(j_idx))
                        flat_idx += 1


fn copy(arg: Array) raises -> Array:
    var res = Array(arg.shape(), False, arg.is_complex())
    execute_copy_raw(
        arg.data(), res.data(), arg.array_shape(), arg.is_complex()
    )
    return res


fn contiguous(arg: Array) raises -> Array:
    var arg_stride = arg.stride()
    var expected_stride = compute_stride(arg.shape())
    if arg.is_complex():
        for i in range(len(expected_stride)):
            expected_stride[i] *= 2
    var is_contiguous = True
    for i in range(len(arg_stride)):
        if arg_stride[i] != expected_stride[i]:
            is_contiguous = False
            break
    var res = arg if is_contiguous else copy(arg)
    return res


fn op_array(
    array_shape: ArrayShape,
    args: List[Array],
    kwargs: List[Array],
    name: String,
    fwd: fn (inout Array, List[Array]) raises -> None,
    jvp: fn (List[Array], List[Array]) raises -> Array,
    vjp: fn (List[Array], Array, Array) raises -> List[Array],
    is_view: Bool = False,
    uew_op: Optional[
        fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ]
    ] = None,
    bew_op: Optional[
        fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ]
    ] = None,
) raises -> Array:
    """
    This operation will setup an array i.e. a node in the background with all its necessary data and
    the functions (forward/jvp/vjp etc.) that act on its direct parent nodes. If any of the parent nodes/args
    point to JIT graph on which caches all the operations, we also always use this graph for the current newly created
    array. The JIT FxGraph is always passed as a reference to the node, so that we can always access the graph.
    """
    var res_arr = Array(array_shape)
    res_arr.set_fwd(fwd)
    res_arr.kwargs_(kwargs)
    res_arr.set_name(name)
    res_arr.is_view_(is_view)

    var requires_grad = False
    var is_complex = False
    for arg in args:
        requires_grad = requires_grad or arg[].requires_grad()
        is_complex = is_complex or arg[].is_complex()
    res_arr.is_complex_(is_complex)
    if requires_grad:
        res_arr.requires_grad_(True)
        res_arr.args_(args)
        res_arr.jvp_(jvp)
        res_arr.vjp_(vjp)

    if uew_op:
        var op = uew_op
        res_arr.append_uew(op.unsafe_take())

    if bew_op:
        var op = bew_op
        res_arr.append_bew(op.unsafe_take())

    # print("Try: registering operation", res_arr.name())

    # go through all args and if any of them points to a graph we take this graph and also let the current node point to this graph, then we register the opertaitaon with the node in the graph
    for arg in args:  # res_arr.args():
        if arg[].has_fxgraph():
            # print("Do: registering operation", res_arr.name())
            var graph = arg[].graph()
            graph[].op_arrayeration(res_arr)
            res_arr.graph_(graph)

            # var arr_shape = res_arr.graph_dual().array_shape()
            # compute_shape(arr_shape, True)

            var res_arr_dual = res_arr.graph_dual()

            # adapt args of dual_arr to catch external args as well
            var adapted_args = res_arr_dual.args()
            for i in range(len(adapted_args)):
                var static_outter_arg = args[i]
                if not static_outter_arg.has_fxgraph():
                    # adapted_args[i] = contiguous(static_outter_arg)
                    # adapted_args[i].graph_(graph)
                    # adapted_args[i].set_name("arg")
                    adapted_args[i] = copy(static_outter_arg)
            res_arr_dual.args_(adapted_args)

            # if node has been visited before and is marked as breakpoint, we need to compute it
            var arr_node = graph[].trace[res_arr.id_in_graph()]
            if arr_node.is_breakpoint:
                _ = res_arr.item(0)

            return res_arr

    fwd(res_arr, args)
    return res_arr


fn setup_shape_and_data(inout curr: Array) raises:
    var array_shape = curr.array_shape()
    compute_shape(array_shape, curr.requires_grad() or curr.has_fxgraph())
    var true_size = array_shape.size() if not curr.is_complex() else 2 * array_shape.size()
    curr.data_(DTypePointer[dtype].alloc(true_size))
    memset_zero(curr.data(), true_size)
    if not curr.requires_grad():
        array_shape.shape_node[].args.clear()
    # print("Setup shape and data", curr.name())


@always_inline
fn execute_inplace_ops_inline(
    inout curr: Array,
    uew: List[
        fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ]
    ],
    bew: List[
        fn (
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ) -> Tuple[
            SIMD[dtype, nelts[dtype]() * 2 // 2],
            SIMD[dtype, nelts[dtype]() * 2 // 2],
        ]
    ],
    inplace_infos: List[InplaceInfo],
    args: List[Array],
    idx: Int,
) raises:
    var simd_data = curr.load[nelts[dtype]() * 2 // 2](idx)
    for i in range(len(inplace_infos)):
        var info = inplace_infos[i]
        if info.type == 0:
            var op = uew[info.idx]
            simd_data = op(simd_data, SIMD[dtype, nelts[dtype]() * 2 // 2]())[0]
        else:
            var op = bew[info.idx]
            var arg2 = args[info.arg_id]
            var arg2_data = arg2.load[nelts[dtype]() * 2 // 2](idx)
            simd_data = op(
                simd_data,
                arg2_data,
                SIMD[dtype, nelts[dtype]() * 2 // 2](),
                SIMD[dtype, nelts[dtype]() * 2 // 2](),
            )[0]
    curr.store[nelts[dtype]() * 2 // 2](idx, simd_data)


fn execute_inplace_ops(inout curr: Array) raises:
    var uew = curr.uew()
    var bew = curr.bew()
    var inplace_infos = curr.inplace_infos()
    var args = curr.args()
    for data_idx in range(0, curr.size(), nelts[dtype]()):
        execute_inplace_ops_inline(
            curr, uew, bew, inplace_infos, args, data_idx
        )


# fn execute_unary_op(inout curr: Array, args: List[Array]) raises:
#     var simd_op = curr.uew()[0]
#     var arg0 = contiguous(args[0])
#     var arg0_data = arg0.data()
#     var curr_data = curr.data()
#     var rest_size = curr.size() % nelts[dtype]()
#     var end = curr.size() - rest_size

#     if curr.is_complex():
#         for i in range(0, end, nelts[dtype]()):
#             var idx_real = i * 2
#             # var idx_imag = idx_real + 1
#             var data0 = arg0_data.load[width = nelts[dtype]() * 2](
#                 idx_real
#             ).deinterleave()
#             var res_deinterleaved = simd_op(data0[0], data0[1])
#             var res = res_deinterleaved[0].interleave(res_deinterleaved[1])
#             curr_data.store[width = 2 * ((nelts[dtype]() * 2) // 2)](
#                 idx_real, res
#             )
#         if rest_size != 0:
#             var rest_simd0_real = SIMD[dtype, nelts[dtype]() * 2 // 2]()
#             var rest_simd0_imag = SIMD[dtype, nelts[dtype]() * 2 // 2]()

#             for i in range(rest_size):
#                 var idx_real = (end + i) * 2
#                 var idx_imag = idx_real + 1
#                 rest_simd0_real[i] = arg0_data.load(idx_real)
#                 rest_simd0_imag[i] = arg0_data.load(idx_imag)
#             var res = simd_op(
#                 rest_simd0_real,
#                 rest_simd0_imag,
#             )
#             for i in range(rest_size):
#                 var idx_real = (end + i) * 2
#                 var idx_imag = idx_real + 1
#                 curr_data.store(idx_real, res[0][i])
#                 curr_data.store(idx_imag, res[1][i])
#     else:
#         for i in range(0, end, nelts[dtype]()):
#             var res = simd_op(
#                 arg0_data.load[width = nelts[dtype]() * 2 // 2](i),
#                 SIMD[dtype, nelts[dtype]() * 2 // 2](0),
#             )[0]
#             curr_data.store[width = nelts[dtype]() * 2 // 2](i, res)

#         # now we vectorize along the last dimesion
#         if rest_size != 0:
#             var rest_simd0 = SIMD[dtype, nelts[dtype]() * 2 // 2]()
#             for i in range(rest_size):
#                 rest_simd0[i] = arg0_data.load(i + end)
#             var res = simd_op(
#                 rest_simd0, SIMD[dtype, nelts[dtype]() * 2 // 2]()
#             )[0]
#             for i in range(end, curr.size()):
#                 curr_data.store(i, res[i - end])

#     _ = arg0


# fn execute_binary_op(inout curr: Array, args: List[Array]) raises:
#     var simd_op = curr.bew()[0]
#     var arg0 = contiguous(args[0])
#     var arg1 = contiguous(args[1])
#     if arg0.ndim() == 1 and arg0.shape()[0] == 1:
#         arg0 = full(arg1.shape()[-1], arg0.load(0))
#     if arg1.ndim() == 1 and arg1.shape()[0] == 1:
#         arg1 = full(arg0.shape()[-1], arg1.load(0))
#     var arg0_data = arg0.data()
#     var arg1_data = arg1.data()
#     var curr_data = curr.data()
#     var rest_size = curr.size() % nelts[dtype]()
#     var end = curr.size() - rest_size

#     if curr.is_complex():
#         for i in range(0, end, nelts[dtype]()):
#             var idx_real = i * 2
#             # var idx_imag = idx_real + 1
#             var data0 = arg0_data.load[width = nelts[dtype]() * 2](
#                 idx_real
#             ).deinterleave()
#             var data1 = arg1_data.load[width = nelts[dtype]() * 2](
#                 idx_real
#             ).deinterleave()
#             var res_deinterleaved = simd_op(
#                 data0[0], data1[0], data0[1], data1[1]
#             )
#             var res = res_deinterleaved[0].interleave(res_deinterleaved[1])
#             curr_data.store[width = 2 * ((nelts[dtype]() * 2) // 2)](
#                 idx_real, res
#             )
#         if rest_size != 0:
#             var rest_simd0_real = SIMD[dtype, nelts[dtype]() * 2 // 2]()
#             var rest_simd0_imag = SIMD[dtype, nelts[dtype]() * 2 // 2]()
#             var rest_simd1_real = SIMD[dtype, nelts[dtype]() * 2 // 2]()
#             var rest_simd1_imag = SIMD[dtype, nelts[dtype]() * 2 // 2]()

#             for i in range(rest_size):
#                 var idx_real = (end + i) * 2
#                 var idx_imag = idx_real + 1
#                 rest_simd0_real[i] = arg0_data.load(idx_real)
#                 rest_simd0_imag[i] = arg0_data.load(idx_imag)
#                 rest_simd1_real[i] = arg1_data.load(idx_real)
#                 rest_simd1_imag[i] = arg1_data.load(idx_imag)
#             var res = simd_op(
#                 rest_simd0_real,
#                 rest_simd1_real,
#                 rest_simd0_imag,
#                 rest_simd1_imag,
#             )
#             for i in range(rest_size):
#                 var idx_real = (end + i) * 2
#                 var idx_imag = idx_real + 1
#                 curr_data.store(idx_real, res[0][i])
#                 curr_data.store(idx_imag, res[1][i])
#     else:
#         for i in range(0, end, nelts[dtype]()):
#             var res = simd_op(
#                 arg0_data.load[width = nelts[dtype]() * 2 // 2](i),
#                 arg1_data.load[width = nelts[dtype]() * 2 // 2](i),
#                 SIMD[dtype, nelts[dtype]() * 2 // 2](0),
#                 SIMD[dtype, nelts[dtype]() * 2 // 2](0),
#             )[0]
#             curr_data.store[width = nelts[dtype]() * 2 // 2](i, res)

#         # now we vectorize along teh last dimesion
#         if rest_size != 0:
#             var rest_simd0 = SIMD[dtype, nelts[dtype]() * 2 // 2]()
#             var rest_simd1 = SIMD[dtype, nelts[dtype]() * 2 // 2]()
#             for i in range(rest_size):
#                 rest_simd0[i] = arg0_data.load(i + end)
#                 rest_simd1[i] = arg1_data.load(i + end)
#             var res = simd_op(
#                 rest_simd0,
#                 rest_simd1,
#                 SIMD[dtype, nelts[dtype]() * 2 // 2](0),
#                 SIMD[dtype, nelts[dtype]() * 2 // 2](0),
#             )[0]
#             for i in range(end, curr.size()):
#                 curr_data.store(i, res[i - end])

#     _ = arg0
#     _ = arg1
#     _ = curr
#     _ = args


# fn register_inplace_op(
#     inout curr: Array,
#     op: Variant[
#         fn (
#             SIMD[dtype, nelts[dtype]() * 2 // 2]
#         ) -> SIMD[dtype, nelts[dtype]() * 2 // 2], fn (
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#         ) -> SIMD[dtype, nelts[dtype]() * 2 // 2]
#     ],
#     right_side_arg: Optional[Array] = None,
# ) raises:
#     if op.isa[
#         fn (
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#         ) -> Tuple[
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#         ]
#     ]():
#         # op is a unary elemtwise operation
#         var _op = op[
#             fn (
#                 SIMD[dtype, nelts[dtype]() * 2 // 2],
#                 SIMD[dtype, nelts[dtype]() * 2 // 2],
#             ) -> Tuple[
#                 SIMD[dtype, nelts[dtype]() * 2 // 2],
#                 SIMD[dtype, nelts[dtype]() * 2 // 2],
#             ]
#         ]
#         curr.append_uew(_op)
#         curr.append_inplace_info(InplaceInfo(0, len(curr.uew()) - 1))
#     elif op.isa[
#         fn (
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#         ) -> Tuple[
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#             SIMD[dtype, nelts[dtype]() * 2 // 2],
#         ]
#     ]():
#         # op is a binary elemtwise operation
#         var op = op[
#             fn (
#                 SIMD[dtype, nelts[dtype]() * 2 // 2],
#                 SIMD[dtype, nelts[dtype]() * 2 // 2],
#                 SIMD[dtype, nelts[dtype]() * 2 // 2],
#                 SIMD[dtype, nelts[dtype]() * 2 // 2],
#             ) -> Tuple[
#                 SIMD[dtype, nelts[dtype]() * 2 // 2],
#                 SIMD[dtype, nelts[dtype]() * 2 // 2],
#             ]
#         ]
#         if not right_side_arg:
#             raise "Error in register_inplace_op: right_side_args is None"
#         curr.append_bew(op)
#         var arg = right_side_arg
#         curr.append_arg(arg.unsafe_take())
#         curr.append_inplace_info(
#             InplaceInfo(1, len(curr.bew()) - 1, len(curr.args()) - 1)
#         )
#     else:
#         raise "Error in register_inplace_op: op is not a valid operation"
