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

from endia import Array, complex, zeros_like
from .utils import fft_c


def fft3d(x: Array) -> Array:
    shape = x.shape()
    rows = shape[0]
    cols = shape[1]
    depth = shape[2]

    if not x.is_complex():
        x = complex(x, zeros_like(x))

    # fft_c always applies a 1d fft along the last axis of the input array, the divions parameter is used to divide the input array into smaller arrays which are later concatenated
    x = fft_c(x, divisions=depth * rows)
    x = permute(x, List(2, 0, 1))  # x -> (cols, depth, rows)
    x = fft_c(x, divisions=depth * cols)
    x = permute(x, List(2, 0, 1))  # x -> (rows, cols, depth)
    x = fft_c(x, divisions=rows * cols)
    x = permute(x, List(2, 0, 1))  # x -> (depth, rows, cols)

    return x
