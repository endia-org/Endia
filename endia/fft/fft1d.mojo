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
from endia.utils.aliases import dtype

alias pi = 3.14159265358979323846264


def fft1d(x: nd.Array) -> nd.Array:
    # Convert to complex if the input is real
    if not x.is_complex():
        x = nd.complex(x, nd.zeros_like(x))

    n = x.shape()[0]
    if n <= 1:
        return x

    # Split even and odd indices
    even = fft1d(x[0::2])
    odd = fft1d(x[1::2])

    # Compute twiddle factors
    k = nd.arange(0, n // 2)
    real = nd.cos(-2 * pi * k / n)
    imag = nd.sin(-2 * pi * k / n)
    twiddle_factors = nd.complex(real, imag)

    # Combine results
    combined = nd.Array(List(n), is_complex=True)
    combined[: n // 2] = even + twiddle_factors * odd
    combined[n // 2 :] = even - twiddle_factors * odd

    return combined
