from endia import Array
from endia.utils.aliases import dtype, nelts, NA
import math
from endia.functional._utils import (
    setup_shape_and_data,
)
from ._utils import DifferentiableUnaryOp, unary_op_array, execute_unary_op


####-----------------------------------------------------------------------------------------------------------------####
#### Sign Function
####-----------------------------------------------------------------------------------------------------------------####
struct Sign(DifferentiableUnaryOp):
    @staticmethod
    fn fwd(arg0: Array) raises -> Array:
        """Computes the sign function of the input array element-wise.

        Args:
            arg0: The input array.

        Returns:
            An array containing the sign function of each element in the input array.

        #### Examples:
        ```python
        a = Array([[1, 2], [3, 4]])
        result = sign(a)
        print(result)
        ```

        #### Note:
        This function supports:
        - Complex valued arguments.
        """
        return unary_op_array(
            arg0, "sign", Sign.__call__, Sign.jvp, Sign.vjp, Sign.unary_simd_op
        )

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        return default_vjp(primals, grad, out)

    @staticmethod
    fn unary_simd_op(
        arg0_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg0_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
    ) -> Tuple[
        SIMD[dtype, nelts[dtype]() * 2 // 2],
        SIMD[dtype, nelts[dtype]() * 2 // 2],
    ]:
        """
        Low-level function to compute the sign function of a complex number represented as SIMD vectors.

        Args:
            arg0_real: The real part of the complex number.
            arg0_imag: The imaginary part of the complex number.

        Returns:
            The real and imaginary parts of the sign function of the complex number as a tuple.
        """
        var norm = math.sqrt(arg0_real * arg0_real + arg0_imag * arg0_imag)
        var real = arg0_real / norm
        var imag = arg0_imag / norm
        return real, imag

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """Performs the forward pass for element-wise sign function computation of an array.

        Computes the sign function of each element in the input array and stores the result in the current array.
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


fn sign(arg0: Array) raises -> Array:
    """Computes the sign function of the input array element-wise.

    Args:
        arg0: The input array.

    Returns:
        An array containing the sign function of each element in the input array.

    #### Examples:
    ```python
    a = Array([[1, 2], [3, 4]])
    result = sign(a)
    print(result)
    ```

    #### Note:
    This function supports:
    - Complex valued arguments.
    """
    return Sign.fwd(arg0)
