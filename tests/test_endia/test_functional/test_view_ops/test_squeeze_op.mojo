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


def run_test_squeeze(msg: String = "squeeze"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 1, 40))
    arr_torch = nd.utils.to_torch(arr)

    res = nd.squeeze(arr)
    res_torch = torch.squeeze(arr_torch)

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_squeeze_grad(msg: String = "squeeze_grad"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(2, 1, 40), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)

    res = nd.sum(nd.sin(nd.squeeze(arr)))
    res_torch = torch.sum(torch.sin(torch.squeeze(arr_torch)))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
