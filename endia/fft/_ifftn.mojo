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

from endia import Array, permute, swapaxes
from endia.utils import compute_stride
from endia.functional._utils import is_contiguous, contiguous
from .utils import *  # cooley_tukey_parallel
import math


def ifftn(
    input: Array,
    dims: List[Int] = List[Int](),
    norm: String = "backward",
    out: Optional[Array] = None,
) -> Array:
    """Compute the n-dimensional inverse FFT.

    Args:
        input: The input array.
        dims: The dimensions along which to compute the FFT.
        norm: The normalization mode.
        out: The output array (optional).

    Returns:
        The n-dimensional FFT of the input array.
    """
    var shape = input.shape()
    var ndim = input.ndim()
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

    return cooley_tukey_parallel(
        input, dims, norm, out, True, True, 1.0, normalization_devisor
    )
