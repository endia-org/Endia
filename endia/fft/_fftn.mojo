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


from endia.utils.aliases import dtype, nelts, NA
from endia.functional.unary_ops._utils import DifferentiableUnaryOp
from .utils import DifferentiableFftOp, fft_op_array
from .utils import (
    cooley_tukey_parallel_inplace,
    get_dims_from_encoded_params,
    get_norm_from_encoded_params,
)


struct FFTN(DifferentiableFftOp):
    @staticmethod
    fn fwd(
        arg0: Array,
        dims: List[Int],
        norm: String,
    ) raises -> Array:
        return fft_op_array(
            arg0, "fftn", FFTN.__call__, FFTN.jvp, FFTN.vjp, dims, norm
        )

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        var params = out.meta_data()
        var dims = get_dims_from_encoded_params(params)
        var norm = get_norm_from_encoded_params(params)
        var res = conj(fftn(conj(grad), dims, norm))
        return res

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        setup_shape_and_data(curr)

        var params = curr.meta_data()
        var dims = get_dims_from_encoded_params(params)
        var norm = get_norm_from_encoded_params(params)

        cooley_tukey_parallel_inplace(args[0], curr, dims, norm)


def fftn(
    x: Array,
    dims: List[Int] = List[Int](),
    norm: String = "backward",
) -> Array:
    return FFTN.fwd(x, dims, norm)
