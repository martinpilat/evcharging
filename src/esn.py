import numpy

class ESN:

    def __init__(self, input_dim, output_dim, reservoir_size, alpha):
        self.n_inputs = input_dim
        self.n_outputs = output_dim
        self.n_reservoir = reservoir_size
        self.alpha = alpha

        self.W_in = numpy.random.rand(self.n_reservoir, self.n_inputs+1) * 2 - 1
        self.W_out = numpy.random.rand(self.n_outputs, self.n_reservoir + self.n_inputs + 1) * 2 - 1
        W = numpy.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        W[numpy.random.rand(*W.shape) < 0.9] = 0
        radius = numpy.max(numpy.abs(numpy.linalg.eigvals(W)))
        self.W = W / radius

        self.state = numpy.zeros(self.n_reservoir)

    def update(self, x):
        x_bar = numpy.tanh(self.W_in @ numpy.insert(x, 0, 1.0) + self.W @ self.state)
        self.state = (1 - self.alpha) * self.state + self.alpha * x_bar
        return self.W_out @ numpy.concatenate([numpy.insert(x, 0, 1.0), self.state])

    def reset(self):
        self.state = numpy.zeros(shape=self.state.shape)

if __name__ == '__main__':
    esn = ESN(2, 1, 100, 0.1)
    print(esn.update(numpy.array([[1, 1]])))
    print(esn.update(numpy.array([[1, 1]])))
    esn.reset()
    print(esn.update(numpy.array([[1, 1]])))