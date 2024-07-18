from endia import Array
from endia.utils import (
    setup_array_shape,
    array_shape_to_list,
    list_to_array_shape,
)
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)
from ._utils import DifferentiableViewOp
from endia.functional import reduce_add

####--------------------------------------------------------------------------------------------------------------------####
# Expand/Broadcast
####--------------------------------------------------------------------------------------------------------------------####


struct Expand(DifferentiableViewOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape resulting from broadcasting one array to another.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: Source ArrayShape, target ArrayShape, and axes to ignore during broadcasting.

        #### Constraints:
        - The number of dimensions of the source ArrayShape must be less than or equal to the number of dimensions of the target ArrayShape.
        """
        var scr = args[0]
        var target = args[1]
        var ignore_axes_ = array_shape_to_list(args[2]) if len(
            args
        ) == 3 else List[Int]()
        # adapt axis values
        var ignore_axes = List[Int]()
        for i in range(len(ignore_axes_)):
            if ignore_axes_[i] < 0:
                ignore_axes.append(scr.shape_node[].ndim + ignore_axes_[i])
            else:
                ignore_axes.append(ignore_axes_[i])

        var shape = List[Int]()
        var stride = List[Int]()
        var storage_offset = 0
        var diff = len(target.shape_node[].shape) - len(scr.shape_node[].shape)

        for i in range(diff):
            shape.append(target.shape_node[].shape[i])
            stride.append(0)
        for i in range(len(scr.shape_node[].shape)):
            if list_contains(ignore_axes, i):
                shape.append(scr.shape_node[].shape[i])
                stride.append(scr.shape_node[].stride[i])
                continue
            if scr.shape_node[].shape[i] == target.shape_node[].shape[i + diff]:
                shape.append(scr.shape_node[].shape[i])
                stride.append(scr.shape_node[].stride[i])
            elif scr.shape_node[].shape[i] == 1:
                shape.append(target.shape_node[].shape[i + diff])
                stride.append(0)
            else:
                raise "Error in broadcast_shape_to: Incompatible shapes for broadcasting"

        curr.setup(shape, stride, storage_offset)

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Performs the forward pass for the expand operation. It sets the base of the argument to be the base of the current array and computes the shape of the current array via its dedicated ArraySahpe fwd fucntion.

        Args:
            curr: The current array to store the result (modified in-place).
            args: The array on which the expanded view is created.

        #### Note:
        The information of the shape computation is stored in the ArrayShape object of the curr array.
        """
        var array_shape = curr.array_shape()
        compute_shape(array_shape, curr.requires_grad() or curr.has_fxgraph())

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        """
        Computes the vector-Jacobian product for the expand operation.

        Args:
            primals: A list containing the primal input array.
            grad: The gradient of the output with respect to some scalar function.
            out: The output of the forward pass (unused in this function).

        Returns:
            A list containing the gradient with respect to the input.

        #### Note:
        The vector-Jacobian product for expand is computed by reducing the gradient along the axes that were expanded.
        """
        var stride = out.stride()
        var axis = List[Int]()
        for i in range(grad.ndim()):
            if stride[i] == 0:
                axis.append(i)
        return List(reduce_add(grad, axis))

    @staticmethod
    fn fwd(
        arg0: Array,
        array_shape: ArrayShape,
        ignore_axes: List[Int] = List[Int](),
    ) raises -> Array:
        """
        Expands the input array to the given shape.

        Args:
            arg0: The input array.
            array_shape: The target shape.
            ignore_axes: The axes to ignore during expansion.

        Returns:
            The expanded array.

        #### Constraints:
        - The number of dimensions of the source ArrayShape must be less than or equal to the number of dimensions of the target ArrayShape.
        - The number of axis to ignore must be less than or equal to the number of dimensions of the source ArrayShape.

        #### Note:
        When performing an expand operation in eager mode, the function checks if the shape of the input array is equal to the target shape. If they are equal, the function returns the input array as is. This is done to avoid unnecessary computation.
        """
        # The folliwng code checks if arg0 shape and target_shape are equal, if so and also the arg0 does not have an associated fxgraph, return arg0
        # note: when fxgraph is used, we want to cache the graph but still want to be reactive on the shape changes
        # if not arg0.has_fxgraph():
        var adapted_ignore_axis = List[Int]()
        for i in range(len(ignore_axes)):
            if ignore_axes[i] < 0:
                adapted_ignore_axis.append(arg0.ndim() + ignore_axes[i])
            else:
                adapted_ignore_axis.append(ignore_axes[i])
        if arg0.ndim() == array_shape.ndim():
            var equal = True
            var arg0_shape = arg0.shape()
            var target_shape = array_shape.shape()
            for i in range(arg0.ndim()):
                if list_contains(adapted_ignore_axis, i):
                    continue
                if arg0_shape[i] != target_shape[i]:
                    equal = False
                    break
            if equal:
                return arg0

        var arr_shape = setup_array_shape(
            List(
                arg0.array_shape(),
                array_shape,
                list_to_array_shape(ignore_axes),
            ),
            "brdcst_shape",
            Expand.compute_shape,
        ) if len(ignore_axes) > 0 else setup_array_shape(
            List(arg0.array_shape(), array_shape),
            "brdcst_shape",
            Expand.compute_shape,
        )

        var curr = op_array(
            arr_shape,
            List(arg0),
            NA,
            "brdcst",
            Expand.__call__,
            Expand.jvp,
            Expand.vjp,
            True,
        )
        curr.base_(arg0.base())
        return curr


fn expand(
    arg0: Array, shape: ArrayShape, ignore_axes: List[Int] = List[Int]()
) raises -> Array:
    """
    Expands the input array to the given shape.

    Args:
        arg0: The input array.
        shape: The target shape.
        ignore_axes: The axes to ignore during expansion.

    Returns:
        The expanded array.

    #### Note:
    This function is a wrapper around the expand function with the target shape being the shape of the target array.
    """
    return Expand.fwd(arg0, shape, ignore_axes)


fn expand_as(arg0: Array, arg1: Array) raises -> Array:
    """
    Expands the input array to the shape of the target array.

    Args:
        arg0: The input array.
        arg1: The target array.

    Returns:
        A view on the input array with the shape of the target array.

    #### Note:
    This function is a wrapper around the expand function with the target shape being the shape of the target array.
    """
    return expand(arg0, arg1.array_shape())


fn broadcast_to(arg0: Array, shape: List[Int]) raises -> Array:
    """
    Broadcasts the input array to the given shape.

    Args:
        arg0: The input array.
        shape: The target shape.

    Returns:
        A view on the input array with the target shape.

    #### Note:
    This function is a wrapper around the expand function with the target shape being the shape of the target array.
    """
    return expand(arg0, ArrayShape(shape))
