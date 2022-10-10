import os
import json
import torch
import logging
import configargparse

from tqdm import tqdm
from typing import List, Optional
from torch.utils.data.dataloader import DataLoader
from torch import nn
from transformers import AdamW, get_scheduler, AutoConfig

from dataset import ChinaPeopleDailyNER, id2label, label2id
from model import BertForNER
from utils import setup_tokenizer, set_random_seed, setup_cuda_device


def _get_validated_args(input_args: Optional[List[str]] = None):
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    # CUDA
    parser.add_argument("--no_cuda", action="store_true", help="Not to use CUDA when available")
    # HYPER PARAMETERS
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for each running")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of epochs for model")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    # LOG
    parser.add_argument("--log_dir", type=str, default="logs/test.log", help="Log Directory")
    parser.add_argument("--log2file", action="store_true", help="Output to a .log file")
    # TRANSFORMER TOKENIZER 
    parser.add_argument("--model_name", type=str, default="bert-base-chinese", help="Choose a pretrained base model")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for pretrained models.")
    # MODEL 
    parser.add_argument("--model_weights", type=str, default=None, help="Model weights directory")
    # TRAIN OR EVAL
    parser.add_argument("--do_train", action="store_true", help="Perform training")
    parser.add_argument("--do_eval", action="store_true", help="Perform evaluating on test set")

    args = parser.parse_args(input_args)

    if args.log2file:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
                            datefmt="%m/%d/%Y %H:%M:%S",
                            level=logging.INFO,
                            filename=args.log_dir,
                            filemode='w'
                            )
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
                            datefmt="%m/%d/%Y %H:%M:%S",
                            level=logging.INFO,
                            )

    return args


def evaluate(args, model, dataloader, device):
    model.eval()
    global_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch_x, batch_y = batch
            batch_y.to(device)
            batch_x["input_ids"].to(device)
            batch_x["attention_mask"].to(device)
            batch_x["token_type_ids"].to(device)

            logits = model(batch_x)

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.permute(0, 2, 1), batch_y)
            global_loss += loss.item()

        return global_loss / len(dataloader)


def train(args, model, train_dataloader, dev_dataloader, test_dataloader, device):
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_train_epochs * len(train_dataloader),
    )

    for epoch in range(1, args.num_train_epochs + 1, 1):
        model.train()
        global_loss = 0.
        best_dev_loss = float("inf")
        for i, batch in tqdm(enumerate(train_dataloader),
                             desc=f'Running train for epoch {epoch}',
                             total=len(train_dataloader)):
            batch_x, batch_y = batch
            batch_y.to(device)
            batch_x["input_ids"].to(device)
            batch_x["attention_mask"].to(device)
            batch_x["token_type_ids"].to(device)

            logits = model(batch_x)

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.permute(0, 2, 1), batch_y)
            global_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        logging.info(f'global loss at epoch {epoch}: {global_loss / len(train_dataloader)}')
        logging.info(f'Start Evaluation at epoch {epoch}')

        dev_loss = evaluate(args, model, dev_dataloader, device)
        logging.info(f'dev loss: {dev_loss}')

        if dev_loss < best_dev_loss:
            logging.info(f"new best, new dev loss: {dev_loss} < best_dev_loss: {best_dev_loss}")
            best_dev_loss = dev_loss
            logging.info("saving new weights...")
            torch.save(model.state_dict(), f"epoch_{epoch}_dev_loss_{(100 * dev_loss):.4f}_model_weights.bin")


def main(input_args: Optional[List[str]] = None):
    args = _get_validated_args(input_args)
    device, n_gpu = setup_cuda_device(args.no_cuda)
    logging.info(f"no_cuda {args.no_cuda}")
    logging.info(f"device: {device}, n_gpu: {n_gpu}")
    set_random_seed(args.seed, n_gpu)

    # TODO define model
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = len(id2label)
    model = BertForNER(config)
    model = model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    tokenizer = setup_tokenizer(model_name=args.model_name, cache_dir=args.cache_dir)

    demo_dataset = ChinaPeopleDailyNER(tokenizer, "data/china-people-daily-ner-corpus/example.demo")
    train_dataset = ChinaPeopleDailyNER(tokenizer, "data/china-people-daily-ner-corpus/example.train")
    dev_dataset = ChinaPeopleDailyNER(tokenizer, "data/china-people-daily-ner-corpus/example.dev")
    test_dataset = ChinaPeopleDailyNER(tokenizer, "data/china-people-daily-ner-corpus/example.test")

    demo_dataloader = DataLoader(demo_dataset, batch_size=4, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    if args.do_train:
        train(args, model, train_dataloader, dev_dataloader, test_dataloader, device)

    if args.do_eval:
        if args.model_weights:
            logging.info(f'Loading weights from "{args.model_weights}"...')
            model.load_state_dict(torch.load(args.model_weights))
        else:
            logging.info(f"Loading random weights...")
        test_loss = evaluate(args, model, test_dataloader, device)
        logging.info(f"Test loss: {test_loss}")


if __name__ == "__main__":
    # select gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
