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
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math
from endia.functional._utils import (
    setup_shape_and_data,
)
from ._utils import DifferentiableBinaryOp, execute_binary_op, binary_op_array


####--------------------------------------------------------------------------------------------------------------------####
#### Addition
####--------------------------------------------------------------------------------------------------------------------####


struct Add(DifferentiableBinaryOp):
    @staticmethod
    fn fwd(arg0: Array, arg1: Array) raises -> Array:
        """Adds two arrays element-wise.

        Args:
            arg0: The first input array.
            arg1: The second input array.

        Returns:
            The element-wise sum of arg0 and arg1.

        #### Examples:
        ```python
        a = Array([[1, 2], [3, 4]])
        b = Array([[5, 6], [7, 8]])
        result = add(a, b)
        print(result)
        ```

        #### This function supports
        - Broadcasting.
        - Automatic differentiation (forward and reverse modes).
        - Complex valued arguments.

        """
        return binary_op_array(
            arg0,
            arg1,
            "add",
            Add.__call__,
            Add.jvp,
            Add.vjp,
            Add.binary_simd_op,
        )

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        """
        Compute Jacobian-vector product for array addition.

        Args:
            primals: Primal input arrays.
            tangents: Tangent vectors.

        Returns:
            Array: Jacobian-vector product.

        #### Note:
        Implements forward-mode automatic differentiation for addition.
        The result represents how the output changes with respect to
        infinitesimal changes in the inputs along the directions specified by the tangents.

        #### See Also:
        add_vjp: Reverse-mode autodiff for addition.
        """
        return tangents[0] + tangents[1]

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        """
        Compute vector-Jacobian product for array addition.

        Args:
            primals: Primal input arrays.
            grad: Gradient of the output with respect to some scalar function.
            out: The output of the forward pass.

        Returns:
            List[Array]: Gradients with respect to each input.

        #### Note:
        Implements reverse-mode automatic differentiation for addition.
        Returns arrays with shape zero for inputs that do not require gradients.

        #### See Also:
        add_jvp: Forward-mode autodiff for addition.
        """
        return List(
            grad if primals[0].requires_grad() else Array(0),
            grad if primals[1].requires_grad() else Array(0),
        )

    @staticmethod
    fn binary_simd_op(
        arg0_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg1_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg0_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg1_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
    ) -> Tuple[
        SIMD[dtype, nelts[dtype]() * 2 // 2],
        SIMD[dtype, nelts[dtype]() * 2 // 2],
    ]:
        """
        Low-level function to add two complex numbers represented as SIMD vectors.

        Args:
            arg0_real: The real part of the first complex number.
            arg1_real: The real part of the second complex number.
            arg0_imag: The imaginary part of the first complex number.
            arg1_imag: The imaginary part of the second complex number.

        Returns:
            The real and imaginary parts of the sum of the two complex numbers as a tuple.
        """
        return arg0_real + arg1_real, arg0_imag + arg1_imag

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Adds two arrays element-wise and stores the result in the current array (curr). The function assumes that the shape and data of the args are already set up.
        If the shape and data of the current array (curr) is not set up, the function will compute the shape based on the shapes of the args and set up the data accordingly.

        Args:
            curr: The current array, must be mutable.
            args: The two arrays to add.

        Constraints:
            The two arrays must have broadcastable shapes.
        """
        setup_shape_and_data(curr)
        execute_binary_op(curr, args)


fn add(arg0: Array, arg1: Array) raises -> Array:
    """Adds two arrays element-wise.

    Args:
        arg0: The first input array.
        arg1: The second input array.

    Returns:
        The element-wise sum of arg0 and arg1.

    #### Examples:
    ```python
    a = Array([[1, 2], [3, 4]])
    b = Array([[5, 6], [7, 8]])
    result = add(a, b)
    print(result)
    ```

    #### This function supports
    - Broadcasting.
    - Automatic differentiation (forward and reverse modes).
    - Complex valued arguments.

    """
    return Add.fwd(arg0, arg1)
