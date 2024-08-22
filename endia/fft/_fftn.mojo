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
from .utils import fft_c
import math


def fftn(
    input: Array,
    dims: List[Int] = List[Int](),
    norm: String = "backward",
    out: Optional[Array] = None,
) -> Array:
    """Compute the n-dimensional FFT.

    Args:
        x: The input array.
        dims: The dimensions along which to compute the FFT.
        norm: The normalization mode.
        out: The output array (optional).

    Returns:
        The n-dimensional FFT of the input array.
    """
    var x: Array
    if not input.is_complex():
        x = complex(input, zeros_like(input))
    else:
        x = copy(input)

    if norm == "backward":
        x = x
    elif norm == "forward":
        x = x / x.size()
    elif "ortho":
        x = x / math.sqrt(x.size())
    else:
        raise "fftn: Invalid norm"

    # setup params
    var size = x.size()
    var shape = x.shape()
    var ndim = x.ndim()
    var axes = List[Int]()
    for i in range(ndim):
        axes.append(i)
    var fft_dims = dims if dims.size > 0 else axes
    if dims.size > 0:
        for i in range(fft_dims.size):
            if fft_dims[i] < 0:
                fft_dims[i] = ndim + fft_dims[i]
            if fft_dims[i] < 0 or fft_dims[i] >= ndim:
                raise "Invalid dimension"

    # perform n-dimensional fft
    for dim in fft_dims:
        x = swapaxes(x, dim[], ndim - 1) if dim[] != ndim - 1 else x
        x = fft_c(x, divisions=size // shape[dim[]])
        x = swapaxes(x, ndim - 1, dim[]) if dim[] != ndim - 1 else x

    if out:
        var out_ref = out.value()
        if out_ref.shape() != x.shape():
            raise "fftn: Invalid output shape"
        out_ref[:] = x
        return out_ref

    return x
