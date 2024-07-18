import endia as nd


struct Adagrad:
    var params: List[nd.Array]
    var lr: SIMD[dtype, 1]
    var eps: SIMD[dtype, 1]
    var cache: List[nd.Array]

    fn __init__(
        inout self,
        params: List[nd.Array],
        lr: SIMD[dtype, 1] = 0.01,
        eps: SIMD[dtype, 1] = 1e-8,
    ) raises:
        self.params = params
        self.lr = lr
        self.eps = eps
        self.cache = List[nd.Array]()
        for i in range(len(params)):
            self.cache.append(nd.Array(params[i].shape()))

    fn step(inout self) raises:
        for i in range(len(self.params)):
            self.cache[i] += self.params[i].grad() * self.params[i].grad()
            self.cache[i].clear_args()
            self.params[i] -= (
                self.lr
                * self.params[i].grad()
                / (nd.sqrt(self.cache[i]) + self.eps)
            )
            self.params[i].clear_args()
