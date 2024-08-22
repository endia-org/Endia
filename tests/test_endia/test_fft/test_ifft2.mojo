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

import math
import endia as nd
import time
from python import Python
from endia.fft import ifft2


def ifft2_test():
    var widht = 2**2
    var height = 2**10

    # print("\nInput Width:", widht, " - Height:", height)

    var torch = Python.import_module("torch")

    var shape = List(4, 4, widht, height)
    var x = nd.complex(nd.randn(shape), nd.randn(shape))
    var x_torch = nd.utils.to_torch(x)

    var y = ifft2(x)
    var y_torch = torch.fft.ifft2(x_torch)

    var msg = "ifft2"
    if not nd.utils.is_close(y, y_torch, rtol=1e-6):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
