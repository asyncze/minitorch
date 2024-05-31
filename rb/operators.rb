
$epsilon = 1e-6

# core mathematical operators
    
def mul(x, y)
    return x.to_f * y
end

def id(x)
    return x.to_f
end

def add(x, y)
    return x.to_f + y
end

def neg(x)
    return -x.to_f
end

def lt(x, y)
    return x < y ? 1.0 : 0.0
end

def eg(x, y)
    return x == y ? 1.0 : 0.0
end

def max(x, y)
    return x > y ? x.to_f : y.to_f
end

def is_close(x, y)
    return (x - y).abs.to_f < 1e-2
end

def sigmoid(x)
    return x >= 0 ? 1.0 / (1.0 + Math.exp(-x)) : Math.exp(x) / (1.0 + Math.exp(x)).to_f
end

def relu(x)
    return x >= 0 ? x.to_f : 0.0
end

def log(x)
    return Math.log(x + $epsilon).to_f
end

def exp(x)
    return Math.exp(x).to_f
end

def log_back(x, d)
    return d.to_f / (x + $epsilon)
end

def inv(x)
    return 1.0 / x.to_f
end

def inv_back(x, d)
    return -d / (x * x).to_f
end

def relu_back(x, d)
    return x > 0 ? d.to_f : 0.0
end

def sigmoid_back(x, d)
    return d.to_f * Math.exp(-x) / ((1.0 + Math.exp(-x)) ** 2)
end

# tests

raise Exception unless mul(3, 4) == 12
raise Exception unless id(3) == 3
raise Exception unless add(2, 3) == 5
raise Exception unless neg(3) == -3
raise Exception unless lt(2, 3) == 1
raise Exception unless eg(2, 3) == 0
raise Exception unless max(2, 3) == 3
raise Exception unless !is_close(2, 3)
raise Exception unless is_close(sigmoid(2), 0.88)
raise Exception unless relu(2) == 2
raise Exception unless is_close(log(2), 0.69)
raise Exception unless is_close(exp(2), 7.39)
raise Exception unless is_close(log_back(2, 1), 0.5)
raise Exception unless is_close(inv(2), 0.5)
raise Exception unless is_close(inv_back(2, 1), -0.25)
raise Exception unless relu_back(2, 1) == 1
raise Exception unless is_close(sigmoid_back(2, 1), 0.11)
