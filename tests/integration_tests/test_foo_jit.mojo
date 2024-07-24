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

import endia as torch


def foo(args: List[torch.Array]) -> torch.Array:
    a = args[0]
    b = args[1]
    c = args[2]
    return torch.sum(torch.relu(a @ b + c))


def test_foo_jit():
    foo_jit = torch.jit(torch.value_and_grad(foo))

    a = torch.arange(List(3, 4))
    b = torch.arange(List(4, 5))
    c = torch.arange(List(3, 5))

    var res = foo_jit(List(a, b, c))[List[List[torch.Array]]]
    # print(res[0][0])
    fwd_res = res[0][0]
    a_grad = res[1][0]
    b_grad = res[1][1]
    c_grad = res[1][2]
    # var fwd_res = foo(List(a,b,c))
    # fwd_res.backward()
    # var a_grad = a.grad()
    # var b_grad = b.grad()
    # var c_grad = c.grad()

    print(fwd_res)
    print(a_grad)
    print(b_grad)
    print(c_grad)
