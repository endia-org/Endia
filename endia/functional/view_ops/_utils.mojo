from endia import Array


trait DifferentiableViewOp:
    """
    Trait for binary operations that are differentiable. That mean they define methods for both forward and reverse mode automatic differentiation.
    """

    @staticmethod
    fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
        ...

    @staticmethod
    fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
        ...

    @staticmethod
    fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
        ...

    @staticmethod
    fn __call__(inout curr: Array, args: List[Array]) raises:
        ...
