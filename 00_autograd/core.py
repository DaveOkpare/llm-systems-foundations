import numpy as np


class Value:
    def __init__(self, data, _children=()) -> None:
        self.data = np.asarray(data)
        self.grad = np.zeros_like(self.data)
        self.prev = set(_children)

    def __repr__(self) -> str:
        formatted = np.array2string(self.data, precision=4)
        return f"Value({formatted})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, _children=(self, other))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, _children=(self, other))

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data / other.data, _children=(self, other))

    def mathmul(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(np.matmul(self.data, other.data), _children=(self, other))

    def sum(self, axis=None, keepdims=False):
        result = self.data.sum(axis=axis, keepdims=keepdims)
        return Value(result, _children=(self,))

    def mean(self, axis=None, keepdims=False):
        result = self.data.mean(axis=axis, keepdims=keepdims)
        return Value(result, _children=(self,))

    def relu(self):
        result = np.maximum(0, self.data)
        return Value(result, _children=(self,))

    def log(self, base=None):
        if base is None:
            result = np.log(self.data)
        else:
            result = np.log(self.data) / np.log(base)
        return Value(result, _children=(self,))

    def exp(self):
        result = np.exp(self.data)
        return Value(result, _children=(self,))

    def softmax(self):
        exp_vals = self.exp()
        return exp_vals / exp_vals.sum()
