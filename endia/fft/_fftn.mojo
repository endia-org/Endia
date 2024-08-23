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
from .utils import cooley_tukey_parallel
import math


def fftn(
    input: Array,
    dims: List[Int] = List[Int](),
    norm: String = "backward",
    out: Optional[Array] = None,
) -> Array:
    """Compute the n-dimensional FFT.

    Args:
        input: The input array.
        dims: The dimensions along which to compute the FFT.
        norm: The normalization mode.
        out: The output array (optional).

    Returns:
        The n-dimensional FFT of the input array.
    """
    return cooley_tukey_parallel(input, dims, norm, out)
