import matplotlib.pyplot as plt
import numpy as np
import json
from pprint import pprint

with open("exp3.json") as fd:
    data = json.load(fd)

chis = [(ejk if np.isscalar(ejk) else ejk[-1])
        for ek in zip(*data["chis"])
        for ejk in ek]

pprint(data["chis"])
plt.plot(chis)
plt.show()
