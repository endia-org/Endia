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

from endia import Array, permute
from .utils import fft_c


def fft3d(x: Array) -> Array:
    shape = x.shape()
    planes = shape[0]
    rows = shape[1]
    cols = shape[2]

    if not x.is_complex():
        x = complex(x, zeros_like(x))

    x = fft_c(x, divisions=planes * rows)
    x = permute(x, List(0, 2, 1))  # -> depth, cols, rows
    x = fft_c(x, divisions=planes * cols)
    x = permute(x, List(2, 1, 0))  # -> rows, cols, depth
    x = fft_c(x, divisions=rows * cols)
    x = permute(x, List(2, 0, 1))  # -> depth, rows, cols

    return x
