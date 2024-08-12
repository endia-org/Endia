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


def mean(arg0: Array, axes: List[Int], keepdims: Bool = False) -> Array:
    """
    Computes the mean of the input array along the specified axes.

    Args:
        arg0: The input array.
        axes: The axes along which to compute the mean.
        keepdims: If True, the reduced axes are retained in the output array.

    Returns:
        An array containing the mean of the input array along the specified axes.

    #### Examples:
    ```python
    a = Array([[1, 2], [3, 4]])
    result = mean(a, List(0))
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
    var divisor = (num_elements_arg0 / num_elements_res).cast[dtype]()
    if keepdims:
        return res / divisor
    return squeeze(res / divisor)
