# ===----------------------------------------------------------------------=== #
# Endia 2024
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from endia import Array
from endia.utils import setup_array_shape
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math
from endia.functional._utils import contiguous, is_contiguous
from memory import memcpy

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)
from ._utils import DifferentiableViewOp

####--------------------------------------------------------------------------------------------------------------------####
# Reshape
####--------------------------------------------------------------------------------------------------------------------####


struct Reshape(DifferentiableViewOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after reshaping.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to reshape, and the shape, stride and storage offset of the target ArrayShape encoded in a  single ArrayShape.
        """
        var target = args[1]
        curr.setup(target.shape())

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Performs the forward pass for the reshape operation. It sets the base of the argument to be the base of the current array and computes the shape of the current array via its dedicated ArraySahpe fwd fucntion.

        Args:
            curr: The current array to store the result (modified in-place).
            args: The array on which the reshape view is created.

        #### Note:
        The information of the shape computation is stored in the ArrayShape object of the curr array.

        #### Constraints:
        - The number of elements in the input array must be equal to the number of elements in the target shape.
        """

        var arg = args[0]

        if is_contiguous(arg.array_shape(), arg.is_complex()):
            curr.is_view_(True)
            curr.base_(arg.base())
            var curr_array_shape = curr.array_shape()
            compute_shape(curr_array_shape)
            return
        else:
            setup_shape_and_data(curr)

            if curr.size() != args[0].size():
                raise "The number of elements in the input array must be equal to the number of elements in the target shape."

            var arg_contiguous = contiguous(args[0])
            var arg_contiguous_data = arg_contiguous.data()
            var curr_data = curr.data()
            if curr.is_complex():
                memcpy(curr_data, arg_contiguous_data, curr.size() * 2)
            else:
                memcpy(curr_data, arg_contiguous_data, curr.size())
            _ = arg_contiguous
            _ = curr

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        """
        Computes the vector-Jacobian product for the reshape operation.

        Args:
            primals: A list containing the primal input array.
            grad: The gradient of the output with respect to some scalar function.
            out: The output of the forward pass.

        Returns:
            A list containing the gradient with respect to the input.

        #### Note:
        The vector-Jacobian product for reshape is computed by calling the reshape operation.
        """
        var primal_shape = primals[0].shape()
        return reshape(grad, primal_shape)

    @staticmethod
    fn fwd(arg0: Array, shape: List[Int]) raises -> Array:
        """
        Creates a view of the input array with the given shape.

        Args:
            arg0: The input array.
            shape: The target shape.

        Returns:
            The reshaped array.

        #### Constraints:
        - The number of elements in the input array must be equal to the number of elements in the target shape.
        """
        var arr_shape = setup_array_shape(
            List(arg0.array_shape(), ArrayShape(shape)),
            "reshape_shape",
            Reshape.compute_shape,
        )

        var curr = op_array(
            arr_shape,
            List(arg0),
            NA,
            "reshape",
            Reshape.__call__,
            Reshape.jvp,
            Reshape.vjp,
            # True,
        )
        # curr.base_(arg0.base())
        return curr


fn reshape(arg0: Array, shape: List[Int]) raises -> Array:
    """
    Creates a view of the input array with the given shape.

    Args:
        arg0: The input array.
        shape: The target shape.

    Returns:
        The reshaped array.

    #### Constraints:
    - The number of elements in the input array must be equal to the number of elements in the target shape.
    """
    return Reshape.fwd(arg0, shape)


fn view(arg0: Array, shape: List[Int]) raises -> Array:
    """
    Creates a view of the input array with the given shape.

    Args:
        arg0: The input array.
        shape: The target shape.

    Returns:
        The reshaped array.

    #### Constraints:
    - The number of elements in the input array must be equal to the number of elements in the target shape.

    #### Note:
    This function is a wrapper around the reshape function.
    """
    return reshape(arg0, shape)


fn flatten(arg0: Array) raises -> Array:
    """
    Flattens the input array.

    Args:
        arg0: The input array.

    Returns:
        The flattened array.

    #### Note:
    This function is a wrapper around the reshape function.
    """
    return reshape(arg0, List(-1))
