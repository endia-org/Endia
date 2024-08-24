# # ===----------------------------------------------------------------------=== #
# # Endia 2024
# #
# # Licensed under the Apache License v2.0 with LLVM Exceptions:
# # https://llvm.org/LICENSE.txt
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ===----------------------------------------------------------------------=== #

# from endia import Array, permute, swapaxes
# from endia.utils import compute_stride
# from endia.functional._utils import is_contiguous, contiguous
# from .utils import *  # cooley_tukey_parallel
# import math


# def ifftn(
#     input: Array,
#     dims: List[Int] = List[Int](),
#     norm: String = "backward",
#     out: Optional[Array] = None,
# ) -> Array:
#     """Compute the n-dimensional inverse FFT.

#     Args:
#         input: The input array.
#         dims: The dimensions along which to compute the FFT.
#         norm: The normalization mode.
#         out: The output array (optional).

#     Returns:
#         The n-dimensional FFT of the input array.
#     """
#     var shape = input.shape()
#     var ndim = input.ndim()
#     var axes = List[Int]()
#     for i in range(ndim):
#         axes.append(i)
#     var fft_dims = dims if dims.size > 0 else axes
#     var normalization_devisor = Float32(1.0)
#     for i in range(len(fft_dims)):
#         var dim = fft_dims[i]
#         if dim < 0:
#             dim = ndim + dim
#         if dim < 0 or dim >= ndim:
#             raise "Invalid dimension"
#         normalization_devisor *= shape[dim]

#     return cooley_tukey_parallel(
#         input, dims, norm, out, True, True, 1.0, normalization_devisor
#     )


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


struct IFFTN(DifferentiableFftOp):
    @staticmethod
    fn fwd(
        arg0: Array,
        dims: List[Int],
        norm: String,
    ) raises -> Array:
        """Sets up the Array object for the inverse FFT operation."""
        return fft_op_array(
            arg0, "ifftn", IFFTN.__call__, IFFTN.jvp, IFFTN.vjp, dims, norm
        )

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        """Computes the Jacobian-vector product for the inverse FFT function."""
        return default_jvp(primals, tangents)

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        """Computes the vector-Jacobian product for the inverse FFT function."""
        var params = out.meta_data()
        var dims = get_dims_from_encoded_params(params)
        var norm = get_norm_from_encoded_params(params)
        var res = conj(ifftn(conj(grad), dims, norm))
        return res

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        """Executes the inverse FFT operation inplace."""
        setup_shape_and_data(curr)

        var params = curr.meta_data()
        var dims = get_dims_from_encoded_params(params)
        var norm = get_norm_from_encoded_params(params)

        # cooley_tukey_parallel_inplace(args[0], curr, dims, norm)
        var shape = curr.shape()
        var ndim = curr.ndim()
        var axes = List[Int]()
        for i in range(ndim):
            axes.append(i)
        var fft_dims = dims if dims.size > 0 else axes
        var normalization_devisor = Float32(1.0)
        for i in range(len(fft_dims)):
            var dim = fft_dims[i]
            if dim < 0:
                dim = ndim + dim
            if dim < 0 or dim >= ndim:
                raise "Invalid dimension"
            normalization_devisor *= shape[dim]

        cooley_tukey_parallel_inplace(
            args[0], curr, dims, norm, True, True, 1.0, normalization_devisor
        )


def ifftn(
    x: Array,
    dims: List[Int] = List[Int](),
    norm: String = "backward",
) -> Array:
    """
    Compute the n-dimensional inverse FFT.

    Args:
        x: The input array.
        dims: The dimensions along which to compute the FFT.
        norm: The normalization mode.

    Returns:
        The n-dimensional FFT of the input array.
    """
    return IFFTN.fwd(x, dims, norm)
