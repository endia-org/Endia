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
from endia import fftn


def fftn_test():
    var depth = 2**2
    var width = 2**4
    var height = 2**6

    # print("\nDepth:", depth, " - Width:", width, " - Height:", height)

    var torch = Python.import_module("torch")

    var shape = List(2, 2, depth, width, height)
    var x = nd.complex(nd.randn(shape), nd.randn(shape))
    var x_torch = nd.utils.to_torch(x)

    var y = fftn(x)
    var y_torch = torch.fft.fftn(x_torch)

    var msg = "fftn"
    if not nd.utils.is_close(y, y_torch, rtol=1e-5):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def fft_grad_test():
    var n = 2**12  # power of two
    # print("\nInput Size: ", n)
    var torch = Python.import_module("torch")

    var shape = List(n)
    var x = nd.complex(nd.randn(shape), nd.randn(shape), requires_grad=True)
    var x_torch = nd.utils.to_torch(x).detach().requires_grad_()

    var y = nd.sum(nd.fft(x))
    var y_torch = torch.sum(torch.fft.fft(x_torch))

    y.backward()
    y_torch.real.backward()

    var msg = "fft grad"
    if not nd.utils.is_close(y, y_torch, rtol=1e-5):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def fftn_grad_test():
    var depth = 2**2
    var width = 2**4
    var height = 2**6

    # print("\nDepth:", depth, " - Width:", width, " - Height:", height)

    var torch = Python.import_module("torch")

    var shape = List(2, 2, depth, width, height)
    var x = nd.complex(nd.randn(shape), nd.randn(shape), requires_grad=True)
    var x_torch = nd.utils.to_torch(x).detach().requires_grad_()

    var y = nd.sum(nd.fftn(x))
    var y_torch = torch.sum(torch.fft.fftn(x_torch))

    y.backward()
    y_torch.real.backward()

    var msg = "fftn grad"
    if not nd.utils.is_close(y, y_torch, rtol=1e-5):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
