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
# Detach
####--------------------------------------------------------------------------------------------------------------------####


struct Detach(DifferentiableViewOp):
    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        """
        Computes the shape of an array after detaching.

        Args:
            curr: The ArrayShape to store the result of the computation.
            args: The ArrayShape to detach.
        """
        # Computes the shape of an array after detaching.
        # liek the slice method however we only take the real part of the array
        var arg = args[0]
        curr.setup(arg.shape(), arg.stride(), arg.storage_offset())

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Performs the forward pass for the detach operation. It sets the base of the argument to be the base of the current array and computes the shape of the current array via its dedicated ArraySahpe fwd fucntion.

        Args:
            curr: The current array to store the result (modified in-place).
            args: The array to detach.

        #### Note:
        The information of the shape computation is stored in the ArrayShape object of the curr array.
        """
        curr.base_(args[0].base())
        var array_shape = curr.array_shape()
        compute_shape(array_shape, curr.requires_grad() or curr.has_fxgraph())

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        return default_vjp(primals, grad, out)

    @staticmethod
    fn fwd(arg0: Array) raises -> Array:
        """
        Detaches the input array from the computation graph.

        Args:
            arg0: The input array.

        Returns:
            The detached array.

        #### Note:
        This function is non-differentiable.
        """
        var arr_shape = setup_array_shape(
            List(arg0.array_shape()), "detach_shape", Detach.compute_shape
        )

        var res = op_array(
            arr_shape,
            List(arg0),
            NA,
            "detach",
            Detach.__call__,
            Detach.jvp,
            Detach.vjp,
            True,
        )
        res.requires_grad_(False)
        return res


fn detach(arg0: Array) raises -> Array:
    """
    Detaches the input array from the computation graph.

    Args:
        arg0: The input array.

    Returns:
        The detached array.

    #### Note:
    This function is non-differentiable.
    """
    return Detach.fwd(arg0)
