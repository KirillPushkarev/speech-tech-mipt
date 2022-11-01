from itertools import islice

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf
from torch import no_grad
from tqdm import tqdm

from src.data import ASRDataset, collate_fn
from src.metrics import WER
from src.model import QuartzNetCTC, logger


@hydra.main(config_path="conf", config_name="quartznet_5x5_ru")
def calculate_metrics(conf: omegaconf.DictConfig) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric = WER()

    model = QuartzNetCTC(conf)
    model.eval()
    model.to(device)

    crowd_dataloader, farfield_dataloader = create_dataloaders(conf)

    wer_crowd = compute_metric_on_dataset(crowd_dataloader, metric, model, device, batch_limit=10)
    print(f"WER on crowd test dataset (model with random weights): {wer_crowd}")

    wer_farfield = compute_metric_on_dataset(farfield_dataloader, metric, model, device, batch_limit=10)
    print(f"WER on farfield test dataset (model with random weights): {wer_farfield}")

    ckpt = torch.load(
        "/mnt/c/Users/Kirill/Documents/repos/speech-tech-mipt/week07/asr/data/q5x5_ru_stride_4_crowd_epoch_4_step_9794.ckpt")
    model.load_state_dict(ckpt["state_dict"])
    logger.info("Successfully loaded initial weights")

    model.eval()
    model.to(device)

    crowd_dataloader, farfield_dataloader = create_dataloaders(conf)

    metric = WER()

    wer_crowd = compute_metric_on_dataset(crowd_dataloader, metric, model, device, batch_limit=10)
    print(f"WER on crowd test dataset (model with weights from checkpoint): {wer_crowd}")

    wer_farfield = compute_metric_on_dataset(farfield_dataloader, metric, model, device, batch_limit=10)
    print(f"WER on farfield test dataset (model with weights from checkpoint): {wer_farfield}")


def create_dataloaders(conf):
    crowd_dataloader = torch.utils.data.DataLoader(
        ASRDataset(OmegaConf.merge(conf.val_dataloader.dataset,
                                   OmegaConf.create({"manifest_name": "test_opus/crowd/manifest.jsonl"}))),
        batch_size=16,
        num_workers=conf.val_dataloader.num_workers,
        prefetch_factor=conf.train_dataloader.prefetch_factor,
        collate_fn=collate_fn,
        pin_memory=True
    )
    farfield_dataloader = torch.utils.data.DataLoader(
        ASRDataset(OmegaConf.merge(conf.val_dataloader.dataset,
                                   OmegaConf.create({"manifest_name": "test_opus/farfield/manifest.jsonl"}))),
        batch_size=16,
        num_workers=conf.val_dataloader.num_workers,
        prefetch_factor=conf.train_dataloader.prefetch_factor,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return crowd_dataloader, farfield_dataloader


@no_grad()
def compute_metric_on_dataset(dataloader, metric, model, device, batch_limit=None):
    if batch_limit:
        dataloader = islice(dataloader, batch_limit)

    for batch in tqdm(dataloader):
        features, features_len, targets, target_len = batch
        features = features.to(device)
        features_len = features_len.to(device)

        logprobs, encoded_len, preds = model(features, features_len)
        refs = model.decoder.decode(token_ids=targets, token_ids_length=target_len)
        hyps = model.decoder.decode(
            token_ids=preds, token_ids_length=encoded_len, unique_consecutive=True
        )
        metric.update(references=refs, hypotheses=hyps)

    wer, _, _ = metric.compute()
    metric.reset()

    return wer


if __name__ == "__main__":
    calculate_metrics()
