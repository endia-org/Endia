from endia import Array
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math

from endia.functional._utils import (
    setup_shape_and_data,
)

from endia.functional import log
from ._utils import DifferentiableBinaryOp, execute_binary_op, binary_op_array


####--------------------------------------------------------------------------------------------------------------------####
#### Power Function
####--------------------------------------------------------------------------------------------------------------------####


struct Pow(DifferentiableBinaryOp):
    @staticmethod
    fn fwd(arg0: Array, arg1: Array) raises -> Array:
        """Raises the first array to the power of the second array element-wise.

        Args:
            arg0: The first input array.
            arg1: The second input array.

        Returns:
            The element-wise result of raising arg0 to the power of arg1.

        #### Examples:
        ```python
        a = Array([[1, 2], [3, 4]])
        b = Array([[5, 6], [7, 8]])
        result = pow_to(a, b)
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
            "pow_to",
            Pow.__call__,
            Pow.jvp,
            Pow.vjp,
            Pow.binary_simd_op,
        )

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        """
        Compute Jacobian-vector product for array exponentiation.

        Args:
            primals: Primal input arrays.
            tangents: Tangent vectors.

        Returns:
            Array: Jacobian-vector product.

        #### Note:
        Implements forward-mode automatic differentiation for exponentiation.
        The result represents how the output changes with respect to
        infinitesimal changes in the inputs along the directions specified by the tangents.

        #### See Also:
        pow_vjp: Reverse-mode autodiff for exponentiation.
        """
        return primals[0]

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        """
        Compute vector-Jacobian product for array exponentiation.

        Args:
            primals: Primal input arrays.
            grad: Gradient of the output with respect to some scalar function.
            out: The output of the forward pass.

        Returns:
            List[Array]: Gradients with respect to each input.

        #### Note:
        Implements reverse-mode automatic differentiation for exponentiation.
        Returns arrays with shape zero for inputs that do not require gradients.

        #### See Also:
        pow_jvp: Forward-mode autodiff for exponentiation.
        """
        var lhs_grad = grad * primals[1] * div(out, primals[0]) if primals[
            0
        ].requires_grad() else Array(0)
        var rhs_grad = grad * out * log(primals[0]) if primals[
            1
        ].requires_grad() else Array(0)
        return List(lhs_grad, rhs_grad)

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
        Low-level function to raise a complex number to a complex power represented as SIMD vectors.

        Args:
            arg0_real: The real part of the complex number.
            arg1_real: The real part of the power.
            arg0_imag: The imaginary part of the complex number.
            arg1_imag: The imaginary part of the power.

        Returns:
            The real and imaginary parts of the complex number raised to the complex power as a tuple.
        """
        var log_mag = math.log(
            math.sqrt(arg0_real * arg0_real + arg0_imag * arg0_imag)
        )
        var arg = math.atan2(arg0_imag, arg0_real)
        var u = math.exp(log_mag * arg1_real - arg * arg1_imag)
        var v = log_mag * arg1_imag + arg * arg1_real
        var real_part = u * math.cos(v)
        var imag_part = u * math.sin(v)
        return (real_part, imag_part)

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """
        Raises the first array to the power of the second array element-wise and stores the result in the current array (curr). The function assumes that the shape and data of the args are already set up.
        If the shape and data of the current array (curr) is not set up, the function will compute the shape based on the shapes of the args and set up the data accordingly.

        Args:
            curr: The current array, must be mutable.
            args: The two arrays to raise to the power.

        Constraints:
            The two arrays must have broadcastable shapes.
        """
        setup_shape_and_data(curr)
        execute_binary_op(curr, args)


fn pow_to(arg0: Array, arg1: Array) raises -> Array:
    """Raises the first array to the power of the second array element-wise.

    Args:
        arg0: The first input array.
        arg1: The second input array.

    Returns:
        The element-wise result of raising arg0 to the power of arg1.

    #### Examples:
    ```python
    a = Array([[1, 2], [3, 4]])
    b = Array([[5, 6], [7, 8]])
    result = pow(a, b)
    print(result)
    ```

    #### This function supports
    - Broadcasting.
    - Automatic differentiation (forward and reverse modes).
    - Complex valued arguments.
    """
    return Pow.fwd(arg0, arg1)
