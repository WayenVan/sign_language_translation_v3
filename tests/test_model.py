import hydra
import sys
import torch

sys.path.append("./src")
from csi_slt.modeling_slt.slt import SltConfig, SltModel


def test_slt_model():
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="base_train")
        slt_config = SltConfig(**cfg.model.config)
        slt_model = SltModel(slt_config)
        slt_model.generate(
            input_ids=torch.randint(0, 1000, (1, 10)),
            pixel_values=None,
            pixel_values_length=None,
        )


if __name__ == "__main__":
    test_slt_model()
