# from endia import Array


# trait DifferentiableSpacialOp:
#     @staticmethod
#     fn compute_shape(inout curr: ArrayShape, args: List[ArrayShape]) raises:
#         ...

#     @staticmethod
#     fn fwd(arg0: Array, axis: List[Int]) raises -> Array:
#         ...

#     @staticmethod
#     fn jvp(primals: List[Array], tangents: List[Array]) raises -> Array:
#         ...

#     @staticmethod
#     fn vjp(primals: List[Array], grad: Array, out: Array) raises -> List[Array]:
#         ...

#     @staticmethod
#     fn __call__(inout curr: Array, args: List[Array]) raises:
#         ...
