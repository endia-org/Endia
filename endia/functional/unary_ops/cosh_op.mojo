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
from endia.functional._utils import (
    setup_shape_and_data,
)
from endia.utils.aliases import dtype, nelts, NA
import math

from ._utils import DifferentiableUnaryOp, unary_op_array, execute_unary_op
from endia.functional import sin, sinh


struct CosH(DifferentiableUnaryOp):
    @staticmethod
    fn fwd(arg0: Array) raises -> Array:
        """Computes the hyperbolic cosine of the input array element-wise.

        Args:
            arg0: The input array.

        Returns:
        An array containing the hyperbolic cosine of each element in the input array.

        #### Examples:
        ```python
        a = Array([[1, 2], [3, 4]])
        result = cosh(a)
        print(result)
        ```

        #### Note:
        This function supports:
        - Automatic differentiation (forward and reverse modes).
        - Complex valued arguments.
        """
        return unary_op_array(
            arg0, "cosh", CosH.__call__, CosH.jvp, CosH.vjp, CosH.unary_simd_op
        )

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        """Computes the Jacobian-vector product for the hyperbolic cosine function.

        Implements forward-mode automatic differentiation for the hyperbolic cosine function.

        Args:
            primals: A list containing the primal input array.
            tangents: A list containing the tangent vector.

        Returns:
        The Jacobian-vector product for the hyperbolic cosine function.

        #### Note:
        The Jacobian-vector product for hyperbolic cosine is computed as sinh(x) * dx,
        where x is the primal input and dx is the tangent vector.
        """
        return sinh(primals[0]) * tangents[0]

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        """Computes the vector-Jacobian product for the hyperbolic cosine function.

        Implements reverse-mode automatic differentiation for the hyperbolic cosine function.

        Args:
            primals: A list containing the primal input array.
            grad: The gradient of the output with respect to some scalar function.
            out: The output of the forward pass (unused in this function).

        Returns:
        A list containing the gradient with respect to the input.

        #### Note:
        The vector-Jacobian product for hyperbolic cosine is computed as sinh(x) * grad,
        where x is the primal input and grad is the incoming gradient.
        """
        return sinh(primals[0]) * grad

    @staticmethod
    fn unary_simd_op(
        arg0_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg0_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
    ) -> Tuple[
        SIMD[dtype, nelts[dtype]() * 2 // 2],
        SIMD[dtype, nelts[dtype]() * 2 // 2],
    ]:
        """
        Low-level function to compute the hyperbolic cosine of a complex number represented as SIMD vectors.

        Args:
            arg0_real: The real part of the complex number.
            arg0_imag: The imaginary part of the complex number.

        Returns:
        The real and imaginary parts of the hyperbolic cosine of the complex number as a tuple.
        """
        var real = math.cosh(arg0_real) * math.cos(arg0_imag)
        var imag = math.sinh(arg0_real) * math.sin(arg0_imag)
        return real, imag

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """Performs the forward pass for element-wise hyperbolic cosine computation of an array.

        Computes the hyperbolic cosine of each element in the input array and stores the result in the current array.
        Initializes the current array if not already set up.

        Args:
            curr: The current array to store the result (modified in-place).
            args: A list containing the input array.

        #### Note:
        This function assumes that the shape and data of the args are already set up.
        If the current array (curr) is not initialized, it computes the shape based on the input array and sets up the data accordingly.
        """
        setup_shape_and_data(curr)
        execute_unary_op(curr, args)


fn cosh(arg0: Array) raises -> Array:
    """Computes the hyperbolic cosine of the input array element-wise.

    Args:
        arg0: The input array.

    Returns:
    An array containing the hyperbolic cosine of each element in the input array.

    #### Examples:
    ```python
    a = Array([[1, 2], [3, 4]])
    result = cosh(a)
    print(result)
    ```

    #### Note:
    This function supports:
    - Automatic differentiation (forward and reverse modes).
    - Complex valued arguments.
    """
    return CosH.fwd(arg0)
