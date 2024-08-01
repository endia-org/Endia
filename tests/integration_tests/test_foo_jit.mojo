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

import endia


def foo(args: List[endia.Array]) -> endia.Array:
    a = args[0]
    b = args[1]
    c = args[2]
    return endia.sum(endia.relu(a @ b + c))


def test_foo_jit():
    foo_jit = endia.jit(endia.value_and_grad(foo))

    a = endia.arange(0, 3 * 4).reshape(List(3, 4))
    b = endia.arange(0, 4 * 5).reshape(List(4, 5))
    c = endia.arange(0, 3 * 5).reshape(List(3, 5))

    var res = foo_jit(List(a, b, c))[List[List[endia.Array]]]
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
