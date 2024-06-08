import fire

from train import train
from validate import validate

if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "validate": validate
    })
