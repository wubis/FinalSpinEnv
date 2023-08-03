import numpy as np
from pathlib import Path
from pprint import pprint
import json
root = Path("C:/Users/jyw1/Desktop/SpinW/spinw/logs/experimentlongB")
logs = {}

parts = "chis bics n_bics nn_bics n_nllfs nn_nllfs j1s j2s actions".split()

logs["rewards"] = []
logs["bads"] = []
logs["convergs_n"] = []
logs["convergs_nn"] = []
logs["correct_ends"] = []
logs["ends_j1s"] = []

for p in parts:
    logs[p] = []

for proc in root.glob("*"):
    logs["rewards"].append(np.loadtxt(proc/"rewards.npy").tolist())
    logs["bads"].append(np.loadtxt(proc/"rewards.npy").tolist())
    logs["convergs_n"].append(np.loadtxt(proc/"rewards.npy").tolist())
    logs["convergs_nn"].append(np.loadtxt(proc/"rewards.npy").tolist())
    logs["correct_ends"].append(np.loadtxt(proc/"rewards.npy").tolist())
    logs["ends_j1s"].append(np.loadtxt(proc/"rewards.npy").tolist())

    for p in parts:
        data = []
        for episode in sorted((proc/p).glob("*.npy")):
            data.append(np.loadtxt(episode).tolist())
        logs[p].append(data)

with open("explongB"
          ".json", "w") as fd:
    json.dump(logs, fd, indent=4)

#[pprint(k) for k in zip(logs["rewards"],logs["actions"])]

