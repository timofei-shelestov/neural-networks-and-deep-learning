import numpy as np
import mnist_loader
import network2_vectorized
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2_vectorized.Network([784, 60, 10])
net.default_weight_initializer()

net.SGD(
    training_data,
    30,
    10,
    0.5,
    validation_data=validation_data,
    monitor_validation_acc=True,
    monitor_training_acc=True,
)

# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network2.Network([784, 60, 10])
# net.default_weight_initializer()
#
# net.SGD(
#    training_data,
#    30,
#    10,
#    0.5,
#    evaluation_data=test_data,
#    monitor_evaluation_accuracy=True,
# )
