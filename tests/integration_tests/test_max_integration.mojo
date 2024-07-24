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

from endia import *


def test_max_graph():
    # Define the computation graph
    var a = randu(List(2, 3, 4), requires_grad=True)
    var b = arange(List(4, 5), requires_grad=True)
    var c = arange(List(3, 5), requires_grad=True)
    # var res = squeeze(unsqueeze(sin(a @ b + c), List(0,2)))
    res = ge_zero(a)
    # res = (
    #     squeeze(a) * 2
    # )  # squeeze(reduce_add(relu((a @ b) + c),  List(0,1,2))) * 2
    print(res)
    # a.T()#reduce_add(a,0)#sin(a @ b) + cos(c), List(0))

    # Define trace, args and outputs of the graph
    var trace = top_order(res)
    var args = List(a, b, c)
    var outputs = List[Array]()
    outputs.append(res)

    # create a callable model with MAX
    var callable = build_model(args, outputs, trace)

    # execute_max_graph the model
    for i in range(1):
        var output = execute_model(args, outputs, callable)
        if i % 100 == 0:
            print("JIT Iteration:", i)
            print(output[0])


# def main():
#     test_max_graph()


def foo(args: List[Array]) -> Array:
    a = args[0]
    b = args[1]
    c = args[2]
    return relu(ge_zero(a))
    # return (
    #     a @ b
    # ) + c  # sum(relu(a)) / a.ndim()# squeeze(reduce_add(relu((a @ b) + c),  List(0,1,2)) ) * ones(List(1))


def test_max_integration():
    foo_jit = jit(foo)

    a = randn(List(2, 3, 4))
    b = arange(List(4, 5))
    c = arange(List(3, 5))

    for i in range(1):
        res = foo_jit(List(a, b, c))[Array]
        print(res)

    # a = arange(List(2, 3, 4)) * 10
    # res = foo_jit(List(a))[Array]
    # print(res)

    # a = arange(List(2, 3, 4)) * 0.1
    # res = foo_jit(List(a))[Array]
    # print(res)
