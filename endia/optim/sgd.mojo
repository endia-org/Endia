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


struct SGD:
    var params: List[nd.Array]
    var lr: SIMD[dtype, 1]
    var momentum: SIMD[dtype, 1]
    var weight_decay: SIMD[dtype, 1]
    var velocity: List[nd.Array]

    fn __init__(
        inout self,
        params: List[nd.Array],
        lr: SIMD[dtype, 1] = 0.01,
        momentum: SIMD[dtype, 1] = 0,
        weight_decay: SIMD[dtype, 1] = 0,
    ) raises:
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = List[nd.Array]()
        if momentum > 0:
            for i in range(len(params)):
                self.velocity.append(nd.Array(params[i].shape()))

    fn step(inout self) raises:
        for i in range(len(self.params)):
            if self.momentum > 0 and self.weight_decay > 0:
                self.velocity[i] = self.momentum * self.velocity[
                    i
                ] - self.lr * (
                    self.params[i].grad() + self.weight_decay * self.params[i]
                )
                self.velocity[i].clear_args()
                self.params[i] += self.velocity[i]
            elif self.momentum == 0 and self.weight_decay > 0:
                self.params[i] -= self.lr * (
                    self.params[i].grad() + self.weight_decay * self.params[i]
                )
            elif self.momentum > 0 and self.weight_decay == 0:
                self.velocity[i] = (
                    self.momentum * self.velocity[i]
                    - self.lr * self.params[i].grad()
                )
                self.velocity[i].clear_args()
                self.params[i] += self.velocity[i]
            else:
                self.params[i] -= self.lr * self.params[i].grad()
            self.params[i].clear_args()
