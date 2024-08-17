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

import endia as nd 
from .fft2d import fft2d    


def fft3d(x: nd.Array) -> nd.Array:
    # Apply 1D FFT to rows
    shape = x.shape()
    rows = shape[0]
    cols = shape[1]
    depth = shape[2]

    if not x.is_complex():
        x = nd.complex(x, nd.zeros_like(x))

    for i in range(rows):
        x[i:i+1, :, :] = fft2d(x[i:i+1, :, :].reshape(List(cols, depth))).reshape(List(1, cols, depth))

    for j in range(cols):
        x[:, j:j+1, :] = fft2d(x[:, j:j+1:, :].reshape(List(rows, depth))).reshape(List(rows, 1, depth))

    for k in range(depth):
        x[:, :, k:k+1] = fft2d(x[:, :, k:k+1].reshape(List(rows, cols))).reshape(List(rows, cols, 1))

    return x
