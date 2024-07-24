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
from python import Python


def run_test_conv1d(msg: String = "conv1d"):
    torch = Python.import_module("torch")

    batch_size = 2
    in_channels = 2
    out_channels = 3
    kernel_size = 3
    elements = 6
    stride = 2
    padding = 1
    dilation = 1
    groups = 1

    a = nd.randu(shape=List(batch_size, in_channels, elements))
    kernel = nd.randu(shape=List(out_channels, in_channels, kernel_size))
    bias = nd.randu(shape=List(out_channels))

    a_torch = nd.utils.to_torch(a)
    kernel_torch = nd.utils.to_torch(kernel)
    bias_torch = nd.utils.to_torch(bias)

    res = nd.conv1d(
        a,
        kernel,
        bias,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
    )
    res_torch = torch.nn.functional.conv1d(
        a_torch,
        kernel_torch,
        bias_torch,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
