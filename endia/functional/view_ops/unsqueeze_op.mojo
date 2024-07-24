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
from endia.functional import squeeze

####--------------------------------------------------------------------------------------------------------------------####
# Unsqueeze
####--------------------------------------------------------------------------------------------------------------------####


struct Unsqueeze(DifferentiableViewOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after unsqueezing. This adds dimensions of size 1 along the specified axes.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to unsqueeze, and the axes to unsqueeze along encoded in an ArrayShape.
        """
        var arg = args[0]
        var axis = array_shape_to_list(args[1])
        var shape = arg.shape_node[].shape
        var stride = arg.shape_node[].stride
        var new_stride = List[Int]()
        var new_shape = List[Int]()

        if len(axis) == 0:
            for i in range(len(shape)):
                new_shape.append(shape[i])
                new_stride.append(stride[i])
        else:
            for _ in range(axis[len(axis) - 1] + 1):
                new_shape.append(-1)
                new_stride.append(-1)
            for x in axis:
                new_shape[x[]] = 1
                new_stride[x[]] = -1
            var shape_idx = 0
            for i in range(len(new_shape)):
                if new_shape[i] == -1:
                    new_shape[i] = shape[shape_idx]
                    new_stride[i] = stride[shape_idx]
                    shape_idx += 1
            for i in range(shape_idx, len(shape)):
                new_shape.append(shape[i])
                new_stride.append(stride[i])

            for i in range(len(new_stride) - 2, -1, -1):
                if new_stride[i] == -1:
                    new_stride[i] = new_stride[i + 1]

        curr.setup(new_shape, new_stride, arg.storage_offset())

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Performs the forward pass for the unsqueeze operation. It sets the base of the argument to be the base of the current array and computes the shape of the current array via its dedicated ArraySahpe fwd fucntion.

        Args:
            curr: The current array to store the result (modified in-place).
            args: The array on which the unsqueeze view is created.

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
        Computes the vector-Jacobian product for the unsqueeze operation.

        Args:
            primals: A list containing the primal input array.
            grad: The gradient of the output with respect to some scalar function.
            out: The output of the forward pass (unused in this function).

        Returns:
            A list containing the gradient with respect to the input.

        #### Note:
        The vector-Jacobian product for unsqueeze is computed by squeezing the gradient.
        """
        return List(squeeze(grad))

    @staticmethod
    fn fwd(arg0: Array, axis: ArrayShape) raises -> Array:
        """
        Unsqueezes the input array by adding axes of length 1.

        Args:
            arg0: The input array.
            axis: The axis to unsqueeze.

        Returns:
            The unsqueezed array.
        """
        var arr_shape = setup_array_shape(
            List(arg0.array_shape(), axis),
            "unsqueeze",
            Unsqueeze.compute_shape,
        )

        var curr = op_array(
            arr_shape,
            List(arg0),
            NA,
            "unsqueeze",
            Unsqueeze.__call__,
            Unsqueeze.jvp,
            Unsqueeze.vjp,
            True,
        )
        curr.base_(arg0.base())
        return curr


fn unsqueeze(arg0: Array, axis: ArrayShape) raises -> Array:
    """
    Unsqueezes the input array by adding axes of length 1.

    Args:
        arg0: The input array.
        axis: The axis to unsqueeze.

    Returns:
        The unsqueezed array.
    """
    return Unsqueeze.fwd(arg0, axis)
