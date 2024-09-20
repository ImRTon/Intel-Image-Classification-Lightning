from lightning.pytorch.cli import LightningCLI

from network.model import IntelImageClassificationModel
from network.data import IntelImageClassificationDataModule

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.num_class", "model.num_class", apply_on="instantiate")
        parser.link_arguments("data.class_weights", "model.class_weights", apply_on="instantiate")

def main():
    cli = MyLightningCLI(
        IntelImageClassificationModel, 
        IntelImageClassificationDataModule,
        save_config_kwargs={"overwrite": True}
    )
    
if __name__ == '__main__':
    main()