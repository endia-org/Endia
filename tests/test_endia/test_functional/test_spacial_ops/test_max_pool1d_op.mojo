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


def run_test_max_pool1d(msg: String = "max_pool1d"):
    torch = Python.import_module("torch")

    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    input_tensor = nd.randu(List(2, 2, 10))

    output = nd.max_pool1d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    input_tensor_torch = nd.utils.to_torch(input_tensor)
    output_torch = torch.nn.functional.max_pool1d(
        input_tensor_torch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    if not nd.utils.is_close(output, output_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
