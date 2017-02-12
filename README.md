# TensorflowHelpers
Helper tools I made to work with tensorflow

dependecies:
```tensorflow, pydub, soundfile, pickle```

Examples:

- Minst classifier
```python
from helpers.extractor import *
from helpers.neural_network import *

loader = MinstLoader("data/minst_folder/")
network = ConvNet(loader)

network.Run()
```

- Cifar classifier
```python
from helpers.extractor import *
from helpers.neural_network import *

loader = CifarLoader("data/cifar10", "Grayscale")
network = ConvNet(loader)

network.Run()
```

- Deezer classifier (Notebook: https://github.com/MrNothing/TensorflowHelpers/blob/master/Music%20Classification.ipynb)
```python
from helpers.extractor import *
from helpers.deezer_tools import *
from helpers.neural_network import *

loader = DeezerLoader(sample_size=1/256)
network = ConvNet(loader)

network.Run()
```

- Customize graph:
```python
from helpers.extractor import *
from helpers.neural_network import *

layers = []
layers.append(NNOperation("reshape", [-1, 28, 28, 1]))
layers.append(NNOperation("conv2d", [5, 5, 1, 32]))
layers.append(NNOperation("maxpool2d", 2))
layers.append(NNOperation("conv2d", [5, 5, 32, 64]))
layers.append(NNOperation("maxpool2d", 2))
layers.append(NNOperation("reshape", [-1, 7*7*64]))
layers.append(NNOperation("wx+b", [7*7*64, 1024], [1024]))
layers.append(NNOperation("relu"))
layers.append(NNOperation("dropout", 0.75))
layers.append(NNOperation("wx+b", [1024, 10], [10]))

loader = CifarLoader("data/cifar10", "Grayscale")
network = ConvNet(loader)

network.Run(layers)
```

- Load and Save graph:
```python
from helpers.extractor import *
from helpers.neural_network import *

loader = CifarLoader("data/cifar10", "Grayscale")
network = ConvNet(loader, save_path="graphs/model.ckpt", restore_path="graphs/model.ckpt")

network.Run(layers)
```

- MultiLayer LSTM Example:
```python
from helpers.extractor import *
from helpers.neural_network import *

loader = CifarLoader("inputs/datasets/cifar10/", "RGB")
network = ConvNet(loader, 
                  n_steps=32*3,
                  training_iters=100000, 
                  display_step=1, 
                  learning_rate = 0.001, 
                  batch_size=128)


layers = []
layers.append(LSTMOperation(cells=[1024, 256, 10], n_classes=network.n_classes))

x=tf.placeholder("float", [None, 3*32, 32])
network.Run(x=x, layers=layers, save_path="graphs/Cifar10Graph")
```
- Conv-LSTM graph:
```python
from helpers.extractor import *
from helpers.neural_network import *

loader = CifarLoader("inputs/datasets/cifar10/", "RGB")
network = ConvNet(loader, 
                  training_iters=100000, 
                  display_step=1, 
                  learning_rate = 0.001, 
                  batch_size=128)

layers = []
layers.append(NNOperation("reshape", [-1, 32*3, 32, 1]))
layers.append(NNOperation("conv2d", [3, 3, 1, 32]))
layers.append(NNOperation("maxpool2d", 2)) #16
layers.append(NNOperation("conv2d", [3, 3, 32, 32]))
layers.append(NNOperation("maxpool2d", 2)) #8
layers.append(NNOperation("reshape", [-1, 24*64, 8]))

layers.append(LSTMOperation(cells=[32], n_classes=network.n_classes))
network.Run(layers=layers, save_path="graphs/Cifar10Graph")
```
