#engine.py
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += math.exp(self.data) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data + 1e-7), (self,), 'log')
        def _backward():
            self.grad += (1 / (self.data + 1e-7)) * out.grad
        out._backward = _backward
        return out
    
    @staticmethod
    def softmax(x):
        assert(isinstance(x, list)), "Expected a List of Value objects"
        x_data = [a.data for a in x]
        max_x = max(x_data)
        e_x = [(a-max_x).exp() for a in x]
        sum_ex = sum(e_x)
        return [a / sum_ex for a in e_x]
    
    def __truediv__(self, other): #for softmax
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')
        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad -= (self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Value(1 / (1 + math.exp(-self.data)), (self,), 'sigmoid')
        def _backward():
            s = 1 / (1 + math.exp(-self.data))
            self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward
        return out
