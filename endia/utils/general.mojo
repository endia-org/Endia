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

from endia import Array
from endia.functional import *


fn reset_node_id_recursive(inout curr: Array):
    for arg in curr.args():
        if arg[].node[].id != -1:
            reset_node_id_recursive(arg[])
    curr.id_(-1)


fn top_order_rec(inout curr: Array, inout trace: List[Array]):
    for arg in curr.args():
        if arg[].node[].id == -1:
            top_order_rec(arg[], trace)
    curr.id_(len(trace))
    trace.append(curr)


fn zero_grad_recursive(inout curr: Array):
    for arg in curr.args():
        if arg[].id() == -1:
            zero_grad_recursive(arg[])
    curr.id_(1)
    curr.remove_grad()


fn zero_grad_rec(inout curr: Array):
    reset_node_id_recursive(curr)
    zero_grad_recursive(curr)
    reset_node_id_recursive(curr)


fn remove_grad_recursive(inout curr: Array):
    for arg in curr.args():
        if arg[].id() == -1:
            remove_grad_recursive(arg[])
    curr.id_(1)
    curr.remove_grad()


fn remove_grad_rec(inout curr: Array):
    reset_node_id_recursive(curr)
    remove_grad_recursive(curr)
    reset_node_id_recursive(curr)


@value
struct InplaceInfo:
    var type: Int
    var idx: Int
    var arg_id: Int

    fn __init__(inout self, type: Int, idx: Int, arg_id: Int = -1):
        self.type = type
        self.idx = idx
        self.arg_id = arg_id


fn concat_lists(*lists: List[Int]) -> List[Int]:
    var res = List[Int]()
    for l in lists:
        for i in l[]:
            res.append(i[])
    return res


fn list_contains(list: List[Int], val: Int) -> Bool:
    """
    Checks if a list fo Ints contains a specific value.

    Args:
        list: The list of Ints to check.
        val: The value to check for.

    Returns:
        True if the value is in the list, False otherwise.
    """
    for i in range(len(list)):
        if list[i] == val:
            return True
    return False
