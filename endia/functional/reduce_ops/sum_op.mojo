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


fn sum(arg0: Array) raises -> Array:
    """
    Computes the sum of the input array along all axes.

    Args:
        arg0: The input array.

    Returns:
        An array containing the sum of the input array along all axes.

    #### Examples:
    ```python
     a = Array([[1, 2], [3, 4]])
     result = sum(a)
     print(result)
    ```

    #### Note:
    This function supports:
    - Automatic differentiation (forward and reverse modes).
    - Complex valued arguments.
    """
    var array_shape = arg0.array_shape()
    compute_shape(array_shape, arg0.requires_grad())
    var axis = List[Int]()
    for i in range(arg0.ndim()):
        axis.append(i)
    return squeeze(reduce_add(arg0, axis))
