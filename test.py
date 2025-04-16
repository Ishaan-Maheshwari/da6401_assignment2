from cnn_model import FlexibleCNN

model = FlexibleCNN(
    conv_filters=[64, 64, 64, 64, 64],
    kernel_sizes=[3]*5,
    activation='relu',
    dense_neurons=128,
    num_classes=10
)

model.analyze_model_computation()