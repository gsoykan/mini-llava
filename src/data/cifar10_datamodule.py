from typing import Any

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.cifar10_dataset import Cifar10Dataset


class Cifar10DataModule(LightningDataModule):
    """`LightningDataModule` for CIFAR-10 image-captioning (Stage-1 VLM pretraining).

    Targets are short captions of the form "a photo of a {class}", giving 4-6
    tokens per sample. This makes it much easier to verify connector learning
    than open-ended captioning on Flickr8k.

    CIFAR-10 has no official validation split; "test" is used for both val and test.
    """

    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        lm_model_name: str = "HuggingFaceTB/SmolLM2-360M",
        max_length: int = 16,
        max_train_samples: int | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.batch_size_per_device = batch_size

    def setup(self, stage: str | None = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is not None:
            return

        common = {
            "vision_model_name": self.hparams.vision_model_name,
            "lm_model_name": self.hparams.lm_model_name,
            "max_length": self.hparams.max_length,
        }
        self.data_train = Cifar10Dataset(
            split="train", max_samples=self.hparams.max_train_samples, **common
        )
        # CIFAR-10 has no validation split — use test for both val and test
        self.data_val = Cifar10Dataset(split="test", **common)
        self.data_test = Cifar10Dataset(split="test", **common)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        pass

    def state_dict(self) -> dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass
