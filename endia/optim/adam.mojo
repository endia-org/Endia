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


struct Adam:
    var params: List[nd.Array]
    var lr: SIMD[dtype, 1]
    var beta1: SIMD[dtype, 1]
    var beta2: SIMD[dtype, 1]
    var eps: SIMD[dtype, 1]
    var t: SIMD[dtype, 1]
    var m: List[nd.Array]
    var v: List[nd.Array]

    fn __init__(
        inout self,
        params: List[nd.Array],
        lr: SIMD[dtype, 1] = 0.001,
        beta1: SIMD[dtype, 1] = 0.9,
        beta2: SIMD[dtype, 1] = 0.999,
        eps: SIMD[dtype, 1] = 1e-8,
    ) raises:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = SIMD[dtype, 1](1)
        self.m = List[nd.Array]()
        self.v = List[nd.Array]()
        for i in range(len(params)):
            self.m.append(nd.Array(params[i].shape()))
            self.v.append(nd.Array(params[i].shape()))

    fn step(inout self) raises:
        for i in range(len(self.params)):
            self.m[i] = (
                self.beta1 * self.m[i]
                + (1 - self.beta1) * self.params[i].grad()
            )
            self.m[i].clear_args()
            self.v[i] = (
                self.beta2 * self.v[i]
                + (1 - self.beta2)
                * self.params[i].grad()
                * self.params[i].grad()
            )
            self.v[i].clear_args()
            var m_hat = self.m[i] / (1 - self.beta1**self.t)
            var v_hat = self.v[i] / (1 - self.beta2**self.t)
            self.params[i] -= self.lr * m_hat / (nd.sqrt(v_hat) + self.eps)
            self.params[i].clear_args()
        self.t += 1
