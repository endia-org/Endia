import endia as nd


def viz_example1():
    a = nd.arange(List(2, 3), requires_grad=True)
    b = nd.arange(List(3, 4), requires_grad=True)
    c = nd.arange(List(2, 2, 4), requires_grad=True)

    res = nd.sum(a @ b + c)

    nd.utils.visualize_graph(res, "./assets/example1_graph")
