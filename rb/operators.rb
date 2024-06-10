
$epsilon = 1e-6

# core mathematical operators
    
def mul(x, y)
    return x.to_f * y
end

raise Exception unless mul(2, 3) == 6

def id(x)
    return x.to_f
end

raise Exception unless id(3) == 3

def add(x, y)
    return x.to_f + y
end

raise Exception unless add(2, 3) == 5

def neg(x)
    return -x.to_f
end

raise Exception unless neg(3) == -3

def lt(x, y)
    return x < y ? 1.0 : 0.0
end

raise Exception unless lt(2, 3) == 1

def eq(x, y)
    return x == y ? 1.0 : 0.0
end

raise Exception unless eq(2, 3) == 0

def max(x, y)
    return x > y ? x.to_f : y.to_f
end

raise Exception unless max(2, 3) == 3

def is_close(x, y)
    return (x - y).abs.to_f < 1e-2
end

raise Exception unless !is_close(2, 3)

def sigmoid(x)
    return x >= 0 ? 1.0 / (1.0 + Math.exp(-x)) : Math.exp(x) / (1.0 + Math.exp(x)).to_f
end

raise Exception unless is_close(sigmoid(2), 0.88)

def relu(x)
    return x >= 0 ? x.to_f : 0.0
end

raise Exception unless relu(2) == 2

def log(x)
    return Math.log(x + $epsilon).to_f
end

raise Exception unless is_close(log(2), 0.69)

def exp(x)
    return Math.exp(x).to_f
end

raise Exception unless is_close(exp(2), 7.39)

def log_back(x, d)
    return d.to_f / (x + $epsilon)
end

raise Exception unless is_close(log_back(2, 1), 0.5)

def inv(x)
    return 1.0 / x.to_f
end

raise Exception unless is_close(inv(2), 0.5)

def inv_back(x, d)
    return -d / (x * x).to_f
end

raise Exception unless is_close(inv_back(2, 1), -0.25)

def relu_back(x, d)
    return x > 0 ? d.to_f : 0.0
end

raise Exception unless relu_back(2, 1) == 1

def sigmoid_back(x, d)
    return d.to_f * Math.exp(-x) / ((1.0 + Math.exp(-x)) ** 2)
end

raise Exception unless is_close(sigmoid_back(2, 1), 0.11)

# higher-order functions

def map(fn, ls)
    return ls.map { |x| fn.call(x) }
end

raise Exception unless map(method(:neg), [1, 2, 3]) == [-1, -2, -3]

def neg_list(ls)
    return map(method(:neg), ls)
end

raise Exception unless neg_list([1, 2, 3]) == [-1, -2, -3]

def map2(fn, ls1, ls2)
    return ls1.zip(ls2).map { |x, y| fn.call(x, y) }
end

raise Exception unless map2(method(:add), [1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def add_list(ls1, ls2)
    return map2(method(:add), ls1, ls2)
end

raise Exception unless add_list([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def reduce(fn, start, ls)
    def reduce_fn(ls, fn, start)
        for i in 0...ls.length
            start = fn.call(start, ls[i])
        end
        return start
    end
    return reduce_fn(ls, fn, start)
end

raise Exception unless reduce(method(:mul), 2, [1, 2, 3]) == 12

def sum(ls)
    return reduce(method(:add), 0, ls)
end

raise Exception unless sum([1, 2, 3]) == 6

def prod(ls)
    return reduce(method(:mul), 1, ls)
end

raise Exception unless prod([2, 3, 4]) == 24
