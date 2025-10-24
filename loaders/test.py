from load import load_smd
from dataset import Dataset

smd_train: Dataset = load_smd(group="train")
print(smd_train)

smd_test: Dataset = load_smd(group="test")
print(smd_test)