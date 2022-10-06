#! /usr/bin/env python
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

Trains a simple GPT-2 based disambiguation model.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import collections
import json
import os
import sys
sys.path.append("..")
from tqdm import tqdm as progressbar

import torch
import torch.nn as nn
import transformers

from dataloader import Dataloader, VisualFeatureLoader
from ambiguous_candidate_identifier import AmbiguousCandidateIdentifier
from utils import ambiguous_candidates_evaluation as eval_utils


def evaluate_model(model, loader, batch_size, save_path=None, hidden_test=False):
    results = collections.defaultdict(list)
    num_target_candidates = 0
    num_pred_candidates = 0
    num_overlap_candidates = 0
    for batch in progressbar(loader.get_entire_batch(batch_size)):
        output = model(batch)
        per_instance_prediction = []
        for index, (pred, target) in enumerate(zip(output, batch["gt_label"])):
            nz_pred = torch.nonzero(pred > 0)
            pred_list = nz_pred[:, 0].tolist()
            object_map = batch["object_map"][index]
            pred_list = [object_map[ii] for ii in pred_list]
            # Save results if need be.
            new_instance = {
                "turn_id": batch["turn_id"][index],
                "disambiguation_candidates": pred_list,
            }
            results[batch["dialog_id"][index]].append(new_instance)

    # Restructure results JSON and save.
    results = [
        {
            "dialog_id": dialog_id,
            "predictions": predictions,
        }
        for dialog_id, predictions in results.items()
    ]
    if save_path:
        print(f"Saving: {save_path}")
        with open(save_path, "w") as file_id:
            json.dump(results, file_id)

    report = eval_utils.evaluate_ambiguous_candidates(loader.source_data, results)
    return report["recall"], report["precision"], report["f1"]


def main(args):
    if args["backbone"] == "gpt2":
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    num_added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<USER>", "<SYS>"]}
    )
    tokenizer.truncation_side = "left"

    # Dataloaders.
    feature_loader = VisualFeatureLoader(
        args["visual_feature_path"], args["visual_feature_size"]
    )
    if args["use_gpu"]:
        feature_loader.cuda()
    train_loader = Dataloader(tokenizer, feature_loader, args["train_file"], args)
    val_loader = Dataloader(tokenizer, feature_loader, args["dev_file"], args)
    devtest_loader = Dataloader(tokenizer, feature_loader, args["devtest_file"], args)
    # teststd_loader = Dataloader(
    #     tokenizer, feature_loader, args["teststd_file"], args, hidden_labels=True
    # )

    # Result save path.
    if not os.path.exists(args["result_save_path"]):
        os.makedirs(args["result_save_path"])

    # Model.
    model = AmbiguousCandidateIdentifier(tokenizer, args)
    model.train()
    # Loss function.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args["positive_weight"]]))
    if args["use_gpu"]:
        criterion = criterion.cuda()
    # Prepare optimizer and schedule (linear warmup and decay).
    optimizer = transformers.AdamW(
        model.parameters(), lr=args["learning_rate"], eps=args["adam_epsilon"]
    )

    total_steps = (
        int(train_loader.num_instances / args["batch_size"] * args["num_epochs"]) + 1
    )
    num_iters_epoch = train_loader.num_instances // args["batch_size"]
    num_iters_epoch_float = train_loader.num_instances / args["batch_size"]
    next_eval_iter = 0
    num_iters = 0
    best_performance = {"dev": 0.0}
    total_loss = None
    while True:
        model.zero_grad()

        epoch = num_iters / (float(train_loader.num_instances) / args["batch_size"])
        batch = train_loader.get_random_batch(args["batch_size"])
        output = model(batch)

        # Compute criterion per instance.
        per_instance_loss = []
        for pred, target in zip(output, batch["gt_label"]):
            per_instance_loss.append(criterion(pred.view(-1, 1), target.view(-1, 1)))
        loss = sum(per_instance_loss)
        loss.backward()
        optimizer.step()

        loss_float = float(loss.float().item())
        if total_loss:
            total_loss = 0.95 * total_loss + 0.05 * loss_float
        else:
            total_loss = loss_float

        if num_iters % 100 == 0:
            print(f"[Ep: {epoch:.2f}][Loss: {total_loss:.2f}]")

        # Evaluate_model every epoch.
        if num_iters == next_eval_iter:
            model.eval()
            print("Evaluating ..")
            # Get dev results.
            recall, precision, f1 = evaluate_model(
                model, val_loader, args["batch_size"]
            )
            print(
                f"[dev]  Rec: {recall:.4f}  |  Prec: {precision:.4f}  |  F1: {f1:.4f}"
            )

            # Evaluate on devtest and teststd if better dev performance.
            if best_performance["dev"] < f1:
                best_performance["dev"] = f1
                best_performance["iter_id"] = num_iters
                best_performance["epoch"] = epoch

                # Get devtest results.
                save_path = None
                if args["result_save_path"]:
                    save_path = os.path.join(
                        args["result_save_path"], f"results_devtest_best.json"
                    )
                with torch.no_grad():
                    recall, precision, f1 = evaluate_model(
                        model, devtest_loader, args["batch_size"], save_path
                    )
                    best_performance["devtest"] = f1
                    # Check if performance is the best.
                    print(
                        f"[devtest]  Rec: {recall:.4f}  "
                        f"|  Prec: {precision:.4f}  |  F1: {f1:.4f}"
                    )
                    # # Get teststd predictions.
                    # save_path = None
                    # if args["result_save_path"]:
                    #     save_path = os.path.join(
                    #         args["result_save_path"],
                    #         f"results_teststd_{num_iters}.json",
                    #     )
                    # recall, precision, f1 = evaluate_model(
                    #     model,
                    #     teststd_loader,
                    #     args["batch_size"],
                    #     save_path,
                    #     hidden_test=True,
                    # )
                    # best_performance["teststd"] = f1
                    # print(f"Accuracy [teststd]: {accuracy}")
                    print(f"Current best performance: {best_performance}")
            model.train()

        num_iters += 1
        next_eval_iter = int(int(epoch + 1) * num_iters_epoch_float)
        if epoch > args["num_epochs"]:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True, help="Path to train file")
    parser.add_argument("--dev_file", required=True, help="Path to dev file")
    parser.add_argument("--devtest_file", required=True, help="Path to devtest file")
    parser.add_argument(
        "--teststd_file", required=False, help="Path to public teststd file"
    )
    parser.add_argument(
        "--result_save_path", default=None, help="Path to save devtest results"
    )
    parser.add_argument(
        "--visual_feature_path", default=None, help="Path to visual features"
    )
    parser.add_argument(
        "--visual_feature_size",
        type=int,
        default=516,
        help="Size of the visual features",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="Hidden embed size",
    )
    parser.add_argument(
        "--max_turns", type=int, default=5, help="Number of turns in history"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size")
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum length in utterance"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Linear warmup over warmup_steps"
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-8, help="Eps for Adam optimizer"
    )
    parser.add_argument(
        "--positive_weight",
        type=float,
        default=1.0,
        help="Weight for positive examples for class imbalance",
    )
    parser.add_argument(
        "--backbone",
        required=True,
        choices=["gpt2", "bert"],
        default="bert",
        help="Backbone transformer architecture to train",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true", default=False)
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
