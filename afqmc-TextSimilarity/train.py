import os
import json
import torch
import logging
import configargparse

from tqdm import tqdm
from typing import List, Optional
from torch.utils.data import DataLoader, SequentialSampler
from torch import nn
from transformers import AdamW, get_scheduler

from dataset import AFQMCDataset, AFQMCDatasetNOLABEL
from model import SemanticSimilarityNet
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
    # TRANSFORMER TOKENIZER
    parser.add_argument("--model_name", type=str, default="bert-base-chinese", help="Choose a pretrained base model")
    # MODEL
    parser.add_argument("--model_weights", type=str, default=None, help="Model weights directory")
    # TRAIN OR EVAL
    parser.add_argument("--do_train", action="store_true", help="Perform training")
    parser.add_argument("--do_eval", action="store_true", help="Perform evaluating on test set")

    args = parser.parse_args(input_args)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO,
                        # filename=args.log_dir,
                        # filemode='w'
                        )
    return args


def evaluate(model, dataloader, device):
    model.eval()
    # predicted logits & labels in whole dataloader
    all_logits = []; all_labels = []
    global_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            # TODO: can be simplified by wrapping a function "gather_inputs" 
            batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)
            batch_labels = batch_labels.to(device)

            output_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output_logits, batch_labels)
            global_loss += loss.item()

            # add labels & logits in a batch, to labels & logits in whole dataloader
            all_logits.append(output_logits)
            all_labels.append(batch_labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        predicted_logits, predicted_labels = torch.max(all_logits, dim=1)
        acc = float(((all_labels == predicted_labels).sum() / all_labels.shape[0]))
        return global_loss / len(dataloader), acc


def test(args, model, dataloader, device, tokenizer):
    model.eval()
    # predicted logits & labels in whole dataloader
    all_logits = []; all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            # TODO: can be simplified by wrapping a function "gather_inputs" 
            batch_input_ids, batch_attention_mask, batch_token_type_ids = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)

            output_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)

            # add labels & logits in a batch, to labels & logits in whole dataloader
            all_logits.append(output_logits)

        all_logits = torch.cat(all_logits, dim=0)

        predicted_logits, predicted_labels = torch.max(all_logits, dim=1)
        return predicted_labels


def train(args, model, trainDataLoader, devDataLoader, testDataLoader, device):
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_train_epochs * len(trainDataLoader),
    )

    for epoch in range(1, args.num_train_epochs + 1, 1):
        model.train()
        global_loss = 0.
        best_dev_acc = 0.
        for i, batch in tqdm(enumerate(trainDataLoader),
                             desc=f'Running train for epoch {epoch}',
                             total=len(trainDataLoader)):
            # TODO: can be simplified by wrapping a function "gather_inputs" 
            batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)
            batch_labels = batch_labels.to(device)

            output_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)

            loss_fn = nn.CrossEntropyLoss()
            logging.debug(output_logits.shape)
            logging.debug(batch_labels.shape)
            loss = loss_fn(output_logits, batch_labels)
            global_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        logging.info(f'global loss at epoch {epoch}: {global_loss / len(trainDataLoader)}')
        logging.info(f'Start Evaluation at epoch {epoch}')

        # train_loss, train_acc = evaluate(model, trainDataLoader, device)
        dev_loss, dev_acc = evaluate(model, devDataLoader, device)
        logging.info(f'dev loss: {dev_loss}')
        logging.info(f'dev acc: {dev_acc}')

        if dev_acc > best_dev_acc:
            logging.info(f"new best, dev_acc={dev_acc} > best_dev_acc={best_dev_acc}")
            best_dev_acc = dev_acc

            logging.info("saving new weights...")
            torch.save(model.state_dict(), f"epoch_{epoch}_dev_acc_{(100 * dev_acc):.4f}_model_weights.bin")


def main(input_args: Optional[List[str]] = None):
    args = _get_validated_args(input_args)
    device, n_gpu = setup_cuda_device(args.no_cuda)
    logging.info(f'no_cuda {args.no_cuda}')
    logging.info(f'device: {device}, n_gpu: {n_gpu}')
    set_random_seed(args.seed, n_gpu)

    model = SemanticSimilarityNet(args.model_name)
    model = model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    tokenizer = setup_tokenizer(model_name=args.model_name, cache_dir=None)

    trainDataSet = AFQMCDataset("data/afqmc_public/train.json", tokenizer)
    devDataSet = AFQMCDataset("data/afqmc_public/dev.json", tokenizer)
    testDataSet = AFQMCDatasetNOLABEL("data/afqmc_public/test.json", tokenizer)

    trainDataLoader = DataLoader(trainDataSet, batch_size=args.batch_size, shuffle=True)
    devDataLoader = DataLoader(devDataSet, batch_size=args.batch_size, shuffle=False)
    testDataLoader = DataLoader(testDataSet, batch_size=1, shuffle=False)

    # testDataLoader = DataLoader(testDataSet, sampler=SequentialSampler(range(100)), batch_size=1, shuffle=False)
    # batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = next(iter(trainDataLoader))
    # # torch.size([4, 157]), torch.size([4])
    # logging.debug(f'batch_input_ids shape {batch_input_ids.shape}')
    # logging.debug(f'batch_attention_mask shape {batch_attention_mask.shape}')
    # logging.debug(f'batch_token_type_ids {batch_token_type_ids.shape}')
    # logging.debug(f'batch_labels shape {batch_labels.shape}')
    # # test input and model output here...
    # model_output = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
    # logging.debug(f'model_output shape {model_output.shape}')

    if args.do_train:
        train(args, model, trainDataLoader, devDataLoader, testDataLoader, device)

    if args.do_eval:
        if args.model_weights:
            logging.info(f"Loading weights from \"{args.model_weights}\"...")
            model.load_state_dict(torch.load(args.model_weights))
        else:
            logging.info(f"Loading random weights...")

        predicted_labels = test(args, model, testDataLoader, device, tokenizer)
        predicted_labels = list(map(int, predicted_labels.cpu()))
        # logging.debug(predicted_labels)

        for id, label in enumerate(predicted_labels):
            with open("test.json", 'a') as f:
                json.dump({"id": id, "label": int(label)}, f)
                f.write("\n")


if __name__ == "__main__":
    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
