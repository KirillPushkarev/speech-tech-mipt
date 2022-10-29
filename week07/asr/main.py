import hydra
import omegaconf
import pytorch_lightning as pl
import torch

from src.model import QuartzNetCTC, logger
from src.utils import map_model_state_dict


@hydra.main(config_path="conf", config_name="quartznet_5x5_ru")
def main(conf: omegaconf.DictConfig) -> None:
    model = QuartzNetCTC(conf)

    if conf.model.init_weights:
        ckpt = torch.load(conf.model.init_weights, map_location="cpu")
        model.load_state_dict(map_model_state_dict(ckpt["state_dict"]))
        logger.info("Successfully loaded initial weights")

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(save_dir="logs"), **conf.trainer
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
