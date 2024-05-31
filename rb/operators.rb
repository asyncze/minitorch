
epsilon = 1e-6

# core mathematical operators
    
def mul(x, y)
    return x * y
end

def id(x)
    return x
end

def add(x, y)
    return x + y
end

def neg(x)
    return -x
end

def lt(x, y)
    return x < y ? 1.0 : 0.0
end

# tests

raise Exception unless mul(3, 4) == 12
raise Exception unless id(3) == 3
