from endia import Array
from endia.utils import (
    array_shape_to_list,
    compute_stride,
    setup_array_shape,
    list_to_array_shape,
)
from endia.utils.aliases import dtype, nelts, NA
from algorithm import vectorize, parallelize
import math
from endia.functional._utils import contiguous

from endia.functional._utils import (
    op_array,
    setup_shape_and_data,
)
from ._utils import DifferentiableReduceOp
from endia.functional import expand
from endia.functional import squeeze


def variance(
    arg0: Array,
    axes: List[Int] = List(0),
    unbiased: Bool = True,
    keepdims: Bool = False,
) -> Array:
    """
    Computes the variance of the input array along the specified axes.

    Args:
        arg0: The input array.
        axes: The axes along which to compute the variance.
        unbiased: If True, the variance is computed using the unbiased estimator.
        keepdims: If True, the reduced axes are kept in the result.

    Returns:
        An array containing the variance of the input array along the specified axes.

    #### Examples:
    ```python
    a = Array([[1, 2], [3, 4]])
    result = variance(a, List(0))
    print(result)
    ```

    #### Note:
    This function supports:
    - Automatic differentiation (forward and reverse modes).
    - Complex valued arguments.
    """
    var num_elements_arg0 = arg0.size()
    var res = reduce_add(arg0, axes)
    var num_elements_res = res.size()
    var divisor = (num_elements_arg0 / num_elements_res) - 1 if unbiased else (
        num_elements_arg0 / num_elements_res
    )
    var mean_res = res / (num_elements_arg0 / num_elements_res)
    var diff = arg0 - mean_res
    var diff_squared = diff * diff
    var variance = reduce_add(diff_squared, axes) / divisor
    if keepdims:
        return variance
    return squeeze(variance)
