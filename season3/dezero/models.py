import dezero.functions as F
import dezero.layers as L
from dezero import Layer
from dezero import utils


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        outputs = self.forward(*inputs)
        return utils.plot_dot_graph(outputs, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, output_size in enumerate(fc_output_sizes):
            layer = L.Linear(output_size)
            setattr(self, 'layer' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        y = self.layers[-1](x)
        return y

