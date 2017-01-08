# TensorflowHelpers
Helper tools I use to work with tensorflow

Examples:

- Minst classifier
```python
from helpers.extractor import *
from helpers.deezer_tools import *
from helpers.neural_network import *

loader = MinstLoader("data/minst_folder/")
network = ConvNet(loader)

network.Run()
```

- Cifar classifier
```python
from helpers.extractor import *
from helpers.deezer_tools import *
from helpers.neural_network import *

loader = CifarLoader("data/cifar10", "Grayscale")
network = ConvNet(loader)

network.Run()
```

- Deezer classifier
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
from helpers.deezer_tools import *
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
from helpers.deezer_tools import *
from helpers.neural_network import *

loader = CifarLoader("data/cifar10", "Grayscale")
network = ConvNet(loader, save_path="graphs/tensorflow_inception_graph.pb", restore_path="graphs/tensorflow_inception_graph.pb")

network.Run(layers)
```
