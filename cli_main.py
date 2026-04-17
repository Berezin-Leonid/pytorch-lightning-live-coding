from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningDataModule
from module import LitResnet
import datamodule

def main():
    LightningCLI(LitResnet, LightningDataModule, subclass_mode_data=True)

if __name__ == "__main__":
    main()
