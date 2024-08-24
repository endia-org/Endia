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
from ._ifftn import ifftn


def ifft(
    x: Array,
    dim: Int = -1,
    norm: String = "backward",
) -> Array:
    """
    Compute the n-dimensional inverse FFT.

    Args:
        x: The input array.
        dim: The dimension along which to compute the inverse FFT.
        norm: The normalization mode.

    Returns:
        The n-dimensional inverse FFT of the input array.
    """
    return ifftn(x, dim, norm)
