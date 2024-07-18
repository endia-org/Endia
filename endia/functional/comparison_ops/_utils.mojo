from endia import Array
from endia.functional._utils import (
    contiguous,
    op_array,
    setup_array_shape,
)
from endia.utils import NA, broadcast_shapes


trait ComparisonOp:
    """
    Trait for comparison operations which are non-differentiable.
    """

    @staticmethod
    fn fwd(arg0: Array, arg1: Array) raises -> Array:
        ...

    @staticmethod
    fn comparing_simd_op(
        arg0_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg1_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg0_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg1_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
    ) -> Tuple[
        SIMD[dtype, nelts[dtype]() * 2 // 2],
        SIMD[dtype, nelts[dtype]() * 2 // 2],
    ]:
        ...

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        ...


fn comparison_op_array(
    arg0: Array,
    arg1: Array,
    name: String,
    fwd: fn (inout Array, List[Array]) raises -> None,
    inplace_op: Optional[
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
    var arr_shape = setup_array_shape(
        List(arg0.array_shape(), arg1.array_shape()),
        "broadcast shapes",
        broadcast_shapes,
    )
    if not arg0.has_fxgraph() and not arg1.has_fxgraph():
        compute_shape(arr_shape, arg0.requires_grad() or arg1.requires_grad())
    var args = List(
        expand(arg0, arr_shape),
        expand(arg1, arr_shape),
    )

    return op_array(
        arr_shape,
        args,
        NA,
        name,
        fwd,
        default_jvp,
        default_vjp,
        False,
        None,
        inplace_op,
    )
