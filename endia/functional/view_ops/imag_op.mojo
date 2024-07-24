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
from endia.functional._utils import contiguous

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)
from ._utils import DifferentiableViewOp

####--------------------------------------------------------------------------------------------------------------------####
# Imaginary
####--------------------------------------------------------------------------------------------------------------------####


struct Imag(DifferentiableViewOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of the imaginary part of a complex array.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to compute the imaginary part of.
        """
        # liek the slice method however we only take the real part of the array
        var arg = args[0]
        var shape = arg.shape_node[].shape
        var stride = arg.shape_node[].stride
        var storage_offset = arg.shape_node[].storage_offset + stride[
            len(stride) - 1
        ]
        var new_shape = List[Int]()
        var new_stride = List[Int]()
        for i in range(len(shape)):
            new_shape.append(shape[i])
            new_stride.append(stride[i] * 2)
        curr.setup(new_shape, new_stride, storage_offset)

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        return default_vjp(primals, grad, out)

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Performs the forward pass for the imag operation. It sets the base of the argument to be the base of the current array and computes the shape of the current array via its dedicated ArraySahpe fwd fucntion.

        Args:
            curr: The current array to store the result (modified in-place).
            args: The array on which the imag view is created.

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
            "imag_shape",
            Imag.compute_shape,
        )

        var res = op_array(
            arr_shape,
            List(arg0),
            NA,
            "imag",
            Imag.__call__,
            Imag.jvp,
            Imag.vjp,
            True,
        )
        res.is_complex_(False)
        res.base_(arg0.base())
        return res


fn imag(arg0: Array) raises -> Array:
    """
    Computes the imaginary part of the input array.

    Args:
        arg0: The input array.

    Returns:
        An array containing the imaginary part of the input array.

    #### Examples:
    ```python
    a = Array([[1, 2], [3, 4]])
    result = imag(a)
    print(result)
    ```

    #### Note:
    This function supports:
    - Complex input arrays.
    - Non-differentiable operation.
    """
    return Imag.fwd(arg0)
