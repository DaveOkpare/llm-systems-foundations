import numpy as np


class Value:
    def __init__(self, data, _children=()) -> None:
        self.data = np.asarray(data)
        self.grad = np.zeros_like(self.data)
        self.prev = set(_children)
        self.backward = lambda: None

    def __repr__(self) -> str:
        formatted = np.array2string(self.data, precision=4)
        return f"Value({formatted})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other))

        def backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out.backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.backward = backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, _children=(self, other))

        def backward():
            self.grad += out.grad / other.data
            other.grad += -out.grad * self.data / other.data**2

        out.backward = backward
        return out

    def __pow__(self, value):
        out = Value(np.pow(self.data, value), _children=(self,))

        def backward():
            self.grad += out.grad * (value * self.data ** (value - 1))

        out.backward = backward
        return out

    def mathmul(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = np.matmul(self.data, other.data)
        out = Value(result, _children=(self, other))

        def backward():
            self.grad += np.matmul(out.grad, other.data.T)
            other.grad += np.matmul(self.data.T, out.grad)

        out.backward = backward
        return out

    def sum(self, axis=None, keepdims=False):
        result = self.data.sum(axis=axis, keepdims=keepdims)
        out = Value(result, _children=(self,))

        def backward():
            grad = out.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out.backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        result = self.data.mean(axis=axis, keepdims=keepdims)
        out = Value(result, _children=(self,))

        def backward():
            grad = out.grad / (self.data.shape[axis] if axis else self.data.size)
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out.backward = backward
        return out

    def relu(self):
        result = np.maximum(0, self.data)
        out = Value(result, _children=(self,))

        def backward():
            self.grad += out.grad * (out.data > 0)

        out.backward = backward
        return out

    def log(self, base=None):
        if base is None:
            result = np.log(self.data)
            factor = 1
        else:
            result = np.log(self.data) / np.log(base)
            factor = 1 / np.log(base)
        out = Value(result, _children=(self,))

        def backward():
            self.grad += out.grad * factor / self.data

        out.backward = backward
        return out

    def exp(self):
        result = np.exp(self.data)
        out = Value(result, _children=(self,))

        def backward():
            self.grad += out.grad * result

        out.backward = backward
        return out

    def softmax(self):
        exp_vals = self.exp()
        out = exp_vals / exp_vals.sum()

        def backward():
            jacobian = np.diag(out.data) - np.outer(out.data, out.data)
            self.grad += jacobian @ out.grad

        out.backward = backward
        return out

    def neg(self):
        out = Value(self.data * -1, _children=(self,))

        def backward():
            self.grad += out.grad * -1

        out.backward = backward
        return out
