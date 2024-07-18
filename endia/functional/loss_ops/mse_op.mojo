import endia as nd


@always_inline
fn mse(pred: nd.Array, target: nd.Array) raises -> nd.Array:
    var diff = nd.sub(pred, target)
    return nd.sum((diff * diff)) / pred.size()
