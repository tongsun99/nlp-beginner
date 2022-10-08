import os
import json
import torch
import logging
import configargparse

from tqdm import tqdm
from typing import List, Optional
from torch.utils.data.dataloader import DataLoader
from torch import nn
from transformers import AdamW, get_scheduler

from dataset import ChinaPeopleDailyNER, id2label, label2id
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
    # LOG
    parser.add_argument("--log_dir", type=str, default="test.log", help="Log Directory")
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


def main(input_args: Optional[List[str]] = None):
    args = _get_validated_args(input_args)
    device, n_gpu = setup_cuda_device(args.no_cuda)
    logging.info(f"no_cuda {args.no_cuda}")
    logging.info(f"device: {device}, n_gpu: {n_gpu}")
    set_random_seed(args.seed, n_gpu)

    # TODO define model
    model = 
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
        train(args, model, trainDataLoader, devDataLoader, testDataLoader, device)

    if args.do_eval:
        if args.model_weights:
            logging.info(f'Loading weights from "{args.model_weights}"...')
            model.load_state_dict(torch.load(args.model_weights))
        else:
            logging.info(f"Loading random weights...")

        predicted_labels = test(args, model, testDataLoader, device, tokenizer)
        predicted_labels = list(map(int, predicted_labels.cpu()))

        for id, label in enumerate(predicted_labels):
            with open("test.json", "a") as f:
                json.dump({"id": id, "label": int(label)}, f)
                f.write("\n")


if __name__ == "__main__":
    # select gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
