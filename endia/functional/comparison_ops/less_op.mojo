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
from ._utils import ComparisonOp, comparison_op_array
from endia.functional.binary_ops._utils import execute_binary_op

####--------------------------------------------------------------------------------------------------------------------####
#### Less Operation
####--------------------------------------------------------------------------------------------------------------------####


struct Less(ComparisonOp):
    @staticmethod
    fn fwd(arg0: Array, arg1: Array) raises -> Array:
        return comparison_op_array(
            arg0,
            arg1,
            "less",
            Less.__call__,
            Less.comparing_simd_op,
        )

    @staticmethod
    fn comparing_simd_op(
        arg0_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg1_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg0_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg1_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
    ) -> Tuple[
        SIMD[dtype, nelts[dtype]() * 2 // 2],
        SIMD[dtype, nelts[dtype]() * 2 // 2],
    ]:
        var real = SIMD[dtype, nelts[dtype]() * 2 // 2](0)
        for i in range(nelts[dtype]() * 2 // 2):
            real[i] = arg0_real[i] < arg1_real[i]
        return (real, SIMD[dtype, nelts[dtype]() * 2 // 2](0))

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        setup_shape_and_data(curr)
        execute_binary_op(curr, args)


fn less(arg0: Array, arg1: Array) raises -> Array:
    return Less.fwd(arg0, arg1)
