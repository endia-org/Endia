from endia import Array


alias dtype = DType.float32


fn nelts[dtype: DType]() -> Int:
    return simdwidthof[dtype]() * 2


alias NA = List[Array]()
