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
from endia.functional import sin

####-----------------------------------------------------------------------------------------------------------------####
#### Conj
####-----------------------------------------------------------------------------------------------------------------####


struct Conj(DifferentiableUnaryOp):
    @staticmethod
    fn fwd(arg0: Array) raises -> Array:
        if not arg0.is_complex():
            return arg0
        return unary_op_array(
            arg0, "conj", Conj.__call__, Conj.jvp, Conj.vjp, Conj.unary_simd_op
        )

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        return conj(grad)

    @staticmethod
    fn unary_simd_op(
        arg0_real: SIMD[dtype, nelts[dtype]() * 2 // 2],
        arg0_imag: SIMD[dtype, nelts[dtype]() * 2 // 2],
    ) -> Tuple[
        SIMD[dtype, nelts[dtype]() * 2 // 2],
        SIMD[dtype, nelts[dtype]() * 2 // 2],
    ]:
        return arg0_real, -arg0_imag

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        setup_shape_and_data(curr)
        execute_unary_op(curr, args)


fn conj(arg0: Array) raises -> Array:
    return Conj.fwd(arg0)
