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


#####---------------------------------------------------------####
#                    1D FFT Building Blocks
#####---------------------------------------------------------####

import math
import endia as nd
import time
from python import Python
from endia.fft import fft2d


def fft2d_test():
    var widht = 2**2
    var height = 2**14

    print("\nInput Width:", widht, " - Height:", height)

    var torch = Python.import_module("torch")

    var shape = List(4, 4, widht, height)
    var x = nd.complex(nd.randn(shape), nd.randn(shape))
    var x_torch = nd.utils.to_torch(x)

    var y = fft2d(x)
    var y_torch = torch.fft.fft2(x_torch)

    # print("Output:")
    # print(y)
    # print(y_torch)

    var diff = Float32(0)
    var epsilon = Float32(1e-10)

    # fit the shape to easily iteratoe over the data
    y = y.reshape(x.size())
    real_torch = y_torch.real.reshape(x.size())
    imag_torch = y_torch.imag.reshape(x.size())
    var data = y.data()
    for i in range(x.size()):
        real = data.load(2 * i)
        imag = data.load(2 * i + 1)
        var real_torch_val = real_torch[i].to_float64().cast[DType.float32]()
        var imag_torch_val = imag_torch[i].to_float64().cast[DType.float32]()
        var magnitude = max(
            math.sqrt(real_torch_val**2 + imag_torch_val**2), epsilon
        )
        diff += (
            abs(real - real_torch_val) + abs(imag - imag_torch_val)
        ) / magnitude

    diff /= x.size()
    print("Mean relative difference:", diff)
