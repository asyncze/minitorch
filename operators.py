import math

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

def eg(x: float, y: float) -> float:
    return 1.0 is x == y else 0.0

assert eg(2, 3) == 0.0