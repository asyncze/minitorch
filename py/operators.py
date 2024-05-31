import math

epsilon = 1e-6

# core mathematical operators

def mul(x: float, y: float) -> float:
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -x

def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0

def eg(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    return x if x > 0 else 0.0

def log(x: float) -> float:
    return math.log(x + epsilon)

def exp(x: float) -> float:
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    return d / (x + epsilon)

def inv(x: float) -> float:
    return 1.0 / x

def inv_back(x: float, d: float) -> float:
    return -d / (x * x)

def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0

def sigmoid_back(x: float, d: float) -> float:
    return d * math.exp(-x) / ((1 + math.exp(-x)) ** 2)

assert mul(2, 3) == 6
assert id(3) == 3
assert add(2, 3) == 5
assert neg(3) == -3
assert lt(2, 3) == 1.0
assert eg(2, 3) == 0.0
assert max(2, 3) == 3
assert is_close(2, 3) == 0.0
assert is_close(sigmoid(2), 0.88)
assert relu(2) == 2
assert is_close(log(2), 0.69)
assert is_close(exp(2), 7.39)
assert is_close(log_back(2, 1), 0.5)
assert is_close(inv(2), 0.5)
assert is_close(inv_back(2, 1), -0.25)
assert relu_back(2, 1) == 1
assert is_close(sigmoid_back(2, 1), 0.11)



