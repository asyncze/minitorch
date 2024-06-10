import ctypes

operators = ctypes.CDLL('./operators.so')

# mul
operators._mul.restype = ctypes.c_double
operators._mul.argtypes = [ctypes.c_double, ctypes.c_double]

assert operators._mul(3, 4) == 12
assert operators._mul(-3, 4) == -12

# id
operators._id.restype = ctypes.c_double
operators._id.argtypes = [ctypes.c_double]

assert operators._id(3) == 3
assert operators._id(-3) == -3

# add
operators._add.restype = ctypes.c_double
operators._add.argtypes = [ctypes.c_double, ctypes.c_double]

assert operators._add(3, 4) == 7
assert operators._add(-3, 4) == 1

# neg
operators._neg.restype = ctypes.c_double
operators._neg.argtypes = [ctypes.c_double]

assert operators._neg(3) == -3
assert operators._neg(-3) == 3

# relu
operators._relu.restype = ctypes.c_double
operators._relu.argtypes = [ctypes.c_double]

assert operators._relu(3) == 3
assert operators._relu(-3) == 0

# ...

# map
operators._map.restype = ctypes.POINTER(ctypes.c_double)
operators._map.argtypes = [
    ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int
]

def map_test_helper(fn, x):
    CFuncType = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
    func = CFuncType(fn) # cast fn as CFuncType

    n = len(x)

    list_as_array = (ctypes.c_double * n)(*x)
    result_ptr = operators._map(func, list_as_array, n)

    result = [result_ptr[i] for i in range(n)]

    libc = ctypes.CDLL("libc.dylib")
    libc.free(result_ptr)

    return result

assert map_test_helper(operators._neg, [1, 2, 3]) == [-1, -2, -3]
assert map_test_helper(operators._relu, [1.0, -2.0, 3.0, -4.0, 5.0]) == [1.0, 0.0, 3.0, 0.0, 5.0]




