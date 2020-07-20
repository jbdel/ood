import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LarsonDataset
from models import (
  DeVriesLarsonModelConfig,
  LarsonModelConfig,
)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment_name", default="",
                      help="Path to write checkpoints, results, etc., relative to 'checkpoints/'")
  parser.add_argument("--checkpoint", default="",
                      help="Path to checkpoint to load, relative to 'checkpoints/{args.experiment_name}/'. If empty, starts from scratch.")
  parser.add_argument("--overfit", default=False,
                      help="Whether or not to overfit on the test dataset. Turns off random transforms.")
  parser.add_argument("--model", default="default",
                      help="Model to train (Options: devries|default).")
  parser.add_argument("--num_epochs", type=int,
                      help="Number of epochs to train.")
  parser.add_argument("--num_epochs_per_save", type=int, default=1,
                      help="Number of epochs between saving checkpoints.")
  parser.add_argument("--hint_rate", type=float)
  parser.add_argument("--lmbda", type=float)
  args = parser.parse_args()

  if args.model == "devries":
    model_config = DeVriesLarsonModelConfig(hint_rate=args.hint_rate,
                                            lmbda=args.lmbda)
  else:
    model_config = LarsonModelConfig()

  init_epoch = 0
  num_epochs_per_save = args.num_epochs_per_save

  # Restores settings
  checkpoints_folder = f"checkpoints/{args.experiment_name}"
  checkpoint_filename = f"{checkpoints_folder}/{args.checkpoint}"
  if os.path.isfile(checkpoint_filename):
    print(f"loading: {checkpoint_filename}")

    checkpoint = torch.load(checkpoint_filename)
    init_epoch = checkpoint["init_epoch"]
    model_config.net.load_state_dict(checkpoint["net"])
    model_config.optimizer.load_state_dict(checkpoint["optimizer"])
    model_config.scheduler.load_state_dict(checkpoint["scheduler"])

  # Defines train and test datasets and loaders
  train_dataset_kwargs = {
    "csv_file": "data/train.csv",
    "root_dir": "data",
    "input_transforms": model_config.train_input_transforms,
    "label_transforms": model_config.train_label_transforms,
  }
  test_dataset_kwargs = {
    "csv_file": "data/test.csv",
    "root_dir": "data",
    "input_transforms": model_config.test_input_transforms,
    "label_transforms": model_config.test_label_transforms,
  }
  if args.overfit:
    print(f"Overfitting to test dataset!")
    train_dataset_kwargs = test_dataset_kwargs

  train_dataset = LarsonDataset(**train_dataset_kwargs)
  train_loader = DataLoader(train_dataset,
                            batch_size=16,
                            shuffle=True,
                            num_workers=8)

  test_dataset = LarsonDataset(**test_dataset_kwargs)
  test_loader = DataLoader(test_dataset,
                           batch_size=16,
                           shuffle=False,
                           num_workers=8)

  for epoch in range(init_epoch, init_epoch + args.num_epochs):
    train_start = time.time()

    # Train phase
    running_loss = 0.0
    model_config.net.train()
    for train_iter, sample in enumerate(train_loader, 0):
      inputs_batch, label_batch, _ = sample

      # Reassigns inputs_batch and label_batch to cuda
      cuda_inputs_batch = list(map(lambda x: x.cuda(), inputs_batch))
      cuda_label_batch = label_batch.cuda()

      model_config.optimizer.zero_grad()

      # output_batches might be a single value or a tuple, depending on what
      # model_config.net.__call__ returns
      output_batches = model_config.net(*cuda_inputs_batch)
      loss = model_config.criterion(output_batches, cuda_label_batch)
      loss.backward()
      model_config.optimizer.step()

      running_loss += loss.item()

    eval_start = time.time()

    # Eval phase
    model_config.net.eval()
    for test_iter, sample in enumerate(test_loader, 0):
      inputs_batch, label_batch, metadata_batch = sample

      # Reassigns inputs_batch and label_batch to cuda
      cuda_inputs_batch = list(map(lambda x: x.cuda(), inputs_batch))
      cuda_label_batch = label_batch.cuda()

      output_batches = model_config.net(*cuda_inputs_batch)
      # loss = model_config.criterion(output_batches, cuda_label_batch)

      model_config.update_error_metrics(output_batches,
                                        label_batch,
                                        metadata_batch)

    error_metrics = model_config.error_metrics
    model_config.scheduler.step(error_metrics["mad"])

    error_metric_str = ", ".join([f"{k}: {v}" for k, v in error_metrics.items()])
    print(f"epoch: {epoch}, {error_metric_str}")
    model_config.reset_error_metrics()

    save_start = time.time()

    # Saves settings
    # if (epoch + 1) % num_epochs_per_save == 0:
    #   if not os.path.exists(checkpoints_folder):
    #     os.makedirs(checkpoints_folder)
    #
    #   checkpoint_filename = f"{checkpoints_folder}/model.{epoch}.pth"
    #
    #   print(f"saving: {checkpoint_filename}")
    #
    #   torch.save({
    #     "init_epoch": epoch + 1,
    #     "net": model_config.net.state_dict(),
    #     "optimizer": model_config.optimizer.state_dict(),
    #     "scheduler": model_config.scheduler.state_dict(),
    #   }, checkpoint_filename)


if __name__ == "__main__":
  main()
