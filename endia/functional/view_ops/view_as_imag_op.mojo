from endia import Array
from endia.utils import setup_array_shape
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math
from endia.functional._utils import contiguous

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)
from ._utils import DifferentiableViewOp

####--------------------------------------------------------------------------------------------------------------------####
# View as Imaginary
####--------------------------------------------------------------------------------------------------------------------####


struct ViewAsImag(DifferentiableViewOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after viewing it as the imaginary part of a complex array.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to view as the imaginary part.
        """
        # this simpyl creates a slice shape which merely has a ge_zero size of 2 in the last dimension
        var arg = args[0]
        var shape = arg.shape_node[].shape
        var stride = arg.shape_node[].stride
        var storage_offset = arg.shape_node[].storage_offset + stride[
            len(stride) - 1
        ]
        var new_shape = List[Int]()
        var new_stride = List[Int]()
        for i in range(len(shape) - 1):
            new_shape.append(shape[i])
            new_stride.append(stride[i])
        new_shape.append(shape[len(shape) - 1] // 2)
        new_stride.append(stride[len(stride) - 1] * 2)
        curr.setup(new_shape, new_stride, storage_offset)

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Performs the forward pass for the view_as_imag operation. It sets the base of the argument to be the base of the current array and computes the shape of the current array via its dedicated ArraySahpe fwd fucntion.

        Args:
            curr: The current array to store the result (modified in-place).
            args: The array on which the view_as_imag view is created.

        #### Note:
        The information of the shape computation is stored in the ArrayShape object of the curr array.
        """
        var array_shape = curr.array_shape()
        compute_shape(array_shape, curr.requires_grad() or curr.has_fxgraph())

    @staticmethod
    fn fwd(arg0: Array) raises -> Array:
        """
        Creates a view of the input array as an imaginary array.

        Args:
            arg0: The input array.

        Returns:
            A view of the input array as an imaginary array.

        #### Note:
        This function is non-differentiable.
        """
        var arr_shape = setup_array_shape(
            List(arg0.array_shape()),
            "view_as_imag_shape",
            ViewAsImag.compute_shape,
        )

        var curr = op_array(
            arr_shape,
            List(arg0),
            NA,
            "view_as_imag",
            ViewAsImag.__call__,
            ViewAsImag.jvp,
            ViewAsImag.vjp,
            True,
        )
        curr.base_(arg0.base())
        return curr

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        return default_vjp(primals, grad, out)


fn view_as_imag(arg0: Array) raises -> Array:
    """
    Creates a view of the input array as an imaginary array.

    Args:
        arg0: The input array.

    Returns:
        A view of the input array as an imaginary array.

    #### Note:
    This function is non-differentiable.
    """
    return ViewAsImag.fwd(arg0)
