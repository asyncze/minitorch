
import math

epsilon = 1e-6

# core mathematical operators

def mul(x: float, y: float) -> float:
    return x * y

assert mul(2, 3) == 6

def id(x: float) -> float:
    return x

assert id(3) == 3

def add(x: float, y: float) -> float:
    return x + y

assert add(2, 3) == 5

def neg(x: float) -> float:
    return -x

assert neg(3) == -3

def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0

assert lt(2, 3) == 1.0

def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0

assert eq(2, 3) == 0.0

def max(x: float, y: float) -> float:
    return x if x > y else y

assert max(2, 3) == 3

def is_close(x: float, y: float) -> float:
    return abs(x - y) < 1e-2

assert is_close(2, 3) == 0.0

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

assert is_close(sigmoid(2), 0.88)

def relu(x: float) -> float:
    return x if x > 0 else 0.0

assert relu(2) == 2

def log(x: float) -> float:
    return math.log(x + epsilon)

assert is_close(log(2), 0.69)

def exp(x: float) -> float:
    return math.exp(x)

assert is_close(exp(2), 7.39)

def log_back(x: float, d: float) -> float:
    return d / (x + epsilon)

assert is_close(log_back(2, 1), 0.5)

def inv(x: float) -> float:
    return 1.0 / x

assert is_close(inv(2), 0.5)

def inv_back(x: float, d: float) -> float:
    return -d / (x * x)

assert is_close(inv_back(2, 1), -0.25)

def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0

assert relu_back(2, 1) == 1

def sigmoid_back(x: float, d: float) -> float:
    return d * math.exp(-x) / ((1 + math.exp(-x)) ** 2)

assert is_close(sigmoid_back(2, 1), 0.11)

# higher-order functions

def map(fn):
    return lambda ls: [fn(x) for x in ls]

assert map(neg)([1, 2, 3]) == [-1, -2, -3]

def neg_list(ls):
    return map(neg)(ls)

assert neg_list([1, 2, 3]) == [-1, -2, -3]

def map2(fn):
    return lambda ls1, ls2: [fn(x, y) for x, y in zip(ls1, ls2)]

assert map2(add)([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def add_list(ls1, ls2):
    return map2(add)(ls1, ls2)

assert add_list([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def reduce(fn, start):
    def reduce_fn(ls, fn, start):
        iterator = iter(ls)
        for i in iterator:
            start = fn(start, i)
        return start

    return lambda ls: reduce_fn(ls, fn, start)

assert reduce(mul, 2)([1, 2, 3]) == 12

def sum(ls):
    return reduce(add, 0)(ls)

assert sum([1, 2, 3]) == 6

def prod(ls):
    return reduce(mul, 1)(ls)

assert prod([2, 3, 4]) == 24
