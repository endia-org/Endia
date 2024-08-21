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

    for i in range(rows):
        for j in range(cols):
            x[i : i + 1, j : j + 1, :] = fft_c(x[i : i + 1, j : j + 1, :])
        for k in range(depth):
            x[i : i + 1, :, k : k + 1] = fft_c(x[i : i + 1, :, k : k + 1])

    for j in range(cols):
        for i in range(rows):
            x[i : i + 1, j : j + 1, :] = fft_c(x[i : i + 1, j : j + 1, :])
        for k in range(depth):
            x[:, j : j + 1, k : k + 1] = fft_c(x[:, j : j + 1, k : k + 1])

    for k in range(depth):
        for i in range(rows):
            x[i : i + 1, :, k : k + 1] = fft_c(x[i : i + 1, :, k : k + 1])
        for j in range(cols):
            x[:, j : j + 1, k : k + 1] = fft_c(x[:, j : j + 1, k : k + 1])

    return x
