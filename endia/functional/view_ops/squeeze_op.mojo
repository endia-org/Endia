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
from endia.utils import setup_array_shape, array_shape_to_list
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)
from ._utils import DifferentiableViewOp
from endia.functional import unsqueeze

####--------------------------------------------------------------------------------------------------------------------####
# Squeeze
####--------------------------------------------------------------------------------------------------------------------####


struct Squeeze(DifferentiableViewOp):
    @staticmethod
    fn squeezable_axis(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        var arg = args[0]
        var shape = arg.shape_node[].shape
        var new_shape = List[Int]()
        for i in range(len(shape)):
            if shape[i] == 1:
                new_shape.append(i)
        if len(shape) == len(new_shape):
            _ = new_shape.pop()
        curr.setup(new_shape)

    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after squeezing. This removes all dimensions of size 1.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to squeeze.
        """
        var arg = args[0]
        var _axis = ArrayShape(0)
        Squeeze.squeezable_axis(_axis, List(arg))
        var axis = array_shape_to_list(_axis)
        var shape = arg.shape_node[].shape
        var stride = arg.shape_node[].stride
        var new_shape = List[Int]()
        var new_stride = List[Int]()
        for i in range(len(shape)):
            if not list_contains(axis, i):
                new_shape.append(shape[i])
                new_stride.append(stride[i])
        curr.setup(new_shape, new_stride, arg.storage_offset())
        curr.args_(List(arg, _axis))

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Performs the forward pass for the squeeze operation. It sets the base of the argument to be the base of the current array and computes the shape of the current array via its dedicated ArraySahpe fwd fucntion.

        Args:
            curr: The current array to store the result (modified in-place).
            args: The array on which the squeeze view is created.

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
        Computes the vector-Jacobian product for the squeeze operation.

        Args:
            primals: A list containing the primal input array.
            grad: The gradient of the output with respect to some scalar function.
            out: The output of the forward pass (unused in this function).

        Returns:
            A list containing the gradient with respect to the input.

        #### Note:
        The vector-Jacobian product for squeeze is computed by unsqueezing the gradient along the axes that were squeezed.
        """
        var squeezable_axis = out.array_shape().args()[1]
        return unsqueeze(grad, squeezable_axis)

    @staticmethod
    fn fwd(arg0: Array) raises -> Array:
        """
        Squeezes the input array by removing axes of length 1.

        Args:
            arg0: The input array.

        Returns:
            The squeezed array.
        """
        var arr_shape = setup_array_shape(
            List(arg0.array_shape()),
            "squeeze",
            Squeeze.compute_shape,
        )

        var curr = op_array(
            arr_shape,
            List(arg0),
            NA,
            "squeeze",
            Squeeze.__call__,
            Squeeze.jvp,
            Squeeze.vjp,
            True,
        )
        curr.base_(arg0.base())
        return curr


fn squeeze(arg0: Array) raises -> Array:
    """
    Squeezes the input array by removing axes of length 1.

    Args:
        arg0: The input array.

    Returns:
        The squeezed array.
    """
    return Squeeze.fwd(arg0)
