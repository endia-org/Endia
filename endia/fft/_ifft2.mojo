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

from endia import Array, complex, zeros_like, permute, contiguous
from ._ifftn import ifftn


def ifft2(
    x: Array,
    dims: List[Int] = List(-2, -1),
    norm: String = "backward",
    out: Optional[Array] = None,
) -> Array:
    """Compute the 2-dimensional inverse FFT.

    Args:
        x: The input array.
        dims: The dimensions along which to compute the inverse FFT.
        norm: The normalization mode.

    Returns:
        The 2-dimensional inverse FFT of the input array.
    """
    if len(dims) != 2:
        raise "fft2d: Invalid number of dimensions"
    return ifftn(x, dims, norm, out)
