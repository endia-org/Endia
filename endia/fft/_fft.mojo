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
from ._fftn import fftn


def fft(
    x: Array,
    dim: Int = -1,
    norm: String = "backward",
    out: Optional[Array] = None,
) -> Array:
    """
    Compute the n-dimensional FFT.

    Args:
        x: The input array.
        dim: The dimension along which to compute the FFT.
        norm: The normalization mode.

    Returns:
        The n-dimensional FFT of the input array.
    """
    return fftn(x, dim, norm, out)
