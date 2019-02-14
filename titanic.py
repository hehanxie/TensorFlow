import numpy as np
import tflearn

# download the titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')