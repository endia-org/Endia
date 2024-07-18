from endia import Array
from endia.utils import (
    array_shape_to_slices,
    slices_to_array_shape,
    setup_array_shape,
)
from endia.utils.aliases import dtype, nelts, NA
import math
from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)

from endia.functional import array_slice


####-----------------------------------------------------------------------------------------------------------------####
#### Padding
####-----------------------------------------------------------------------------------------------------------------####


struct Pad:
    @staticmethod
    fn fwd(
        arg0: Array, target_shape: ArrayShape, slices_in_target: List[Slice]
    ) raises -> Array:
        """Pads an array to a target shape.

        Pads the input array to the target shape by copying the input array to the target shape.
        The target shape must be larger than the input array shape.
        The slices in the target shape specify the region where the input array is copied.

        Args:
            arg0: The input array to be padded.
            target_shape: The target shape to pad the input array to.
            slices_in_target: A list of slices specifying the region in the target shape where the input array is copied.

        Returns:
            An array containing the input array padded to the target shape.

        #### Examples:
        ```python
        a = Array([[1, 2], [3, 4]])
        target_shape = ArrayShape([2, 3])
        slices_in_target = [Slice(0, 2), Slice(0, 2)]
        result = pad(a, target_shape, slices_in_target)
        print(result)
        ```

        #### Note:
        This function supports:
        - Automatic differentiation (reverse mode only).
        - Complex valued arguments.
        """
        var arr_shape = setup_array_shape(
            List(
                arg0.array_shape(),
                target_shape,
                slices_to_array_shape(slices_in_target),
            ),
            "pad_shape",
            Pad.padded_shape,
        )

        return op_array(
            arr_shape, List(arg0), NA, "pad", Pad.__call__, default_jvp, Pad.vjp
        )

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        """Computes the vector-Jacobian product for the padding operation.

        Implements reverse-mode automatic differentiation for the padding operation.

        Args:
            primals: A list containing the primal input array and the target shape.
            grad: The gradient of the output with respect to some scalar function.
            out: The output of the forward pass (unused in this function).

        Returns:
            A list containing the gradient with respect to the input.

        #### Note:
        The vector-Jacobian product for padding is computed as the gradient of the output array sliced to the target shape.
        """
        var slices = array_shape_to_slices(out.array_shape().args()[2])
        return List(array_slice(grad, slices))

    @staticmethod
    fn padded_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after padding.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to pad, the target ArrayShape.
        """
        var target_shape = args[1]
        curr.setup(target_shape.shape())

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """Performs the forward pass for padding an array to a target shape.

        Pads the input array to the target shape and stores the result in the current array.
        Initializes the current array if not already set up.

        Args:
            curr: The current array to store the result (modified in-place).
            args: A list containing the input array and the target shape.

        #### Note:
        This function assumes that the shape and data of the args are already set up.
        If the current array (curr) is not initialized, it computes the shape based on the target shape and sets up the data accordingly.
        """
        setup_shape_and_data(curr)
        var arg = args[0]
        var array_shape = curr.array_shape()
        var target_shape = array_shape.args()[1]
        curr.setup_array_shape(target_shape)
        var slices_in_target = array_shape_to_slices(array_shape.args()[2])
        var sliced_curr = array_slice(curr, slices_in_target)

        for i in range(len(target_shape.shape())):
            if arg.shape()[i] != sliced_curr.array_shape().shape()[i]:
                raise "Error in pad: target_shape and sliced_curr shape do not match"

        for i in range(arg.size()):
            sliced_curr.store(i, arg.load(i))


fn pad(
    arg0: Array, target_shape: ArrayShape, slices_in_target: List[Slice]
) raises -> Array:
    """Pads an array to a target shape.

    Pads the input array to the target shape by copying the input array to the target shape.
    The target shape must be larger than the input array shape.
    The slices in the target shape specify the region where the input array is copied.

    Args:
        arg0: The input array to be padded.
        target_shape: The target shape to pad the input array to.
        slices_in_target: A list of slices specifying the region in the target shape where the input array is copied.

    Returns:
        An array containing the input array padded to the target shape.

    #### Examples:
    ```python
    a = Array([[1, 2], [3, 4]])
    target_shape = ArrayShape([2, 3])
    slices_in_target = [Slice(0, 2), Slice(0, 2)]
    result = pad(a, target_shape, slices_in_target)
    print(result)
    ```

    #### Note:
    This function supports:
    - Automatic differentiation (reverse mode only).
    - Complex valued arguments.
    """
    return Pad.fwd(arg0, target_shape, slices_in_target)
