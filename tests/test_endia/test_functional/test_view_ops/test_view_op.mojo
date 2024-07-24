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


def run_test_reshape(msg: String = "reshape"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(30, 4, 5))
    arr_torch = nd.utils.to_torch(arr)

    res = nd.sin(nd.reshape(arr, List(30, 2, 2, 5)))
    res_torch = torch.sin(arr_torch.view(30, 2, 2, 5))

    if not nd.utils.is_close(res, res_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)


def run_test_reshape_grad(msg: String = "reshape_grad"):
    torch = Python.import_module("torch")
    arr = nd.randn(List(30, 4, 5), requires_grad=True)
    arr_torch = nd.utils.to_torch(arr)

    res = nd.sum(nd.sin(nd.reshape(arr, List(30, 2, 2, 5))))
    res_torch = torch.sum(torch.sin(arr_torch.view(30, 2, 2, 5)))

    res.backward()
    res_torch.backward()

    grad = arr.grad()
    grad_torch = arr_torch.grad

    if not nd.utils.is_close(grad, grad_torch):
        print("\033[31mTest failed\033[0m", msg)
    else:
        print("\033[32mTest passed\033[0m", msg)
