"""Script evaluates ambiguous candidates for SIMMC 2.1 using golden labels.

Expected JSON format:

[
    "dialog_id": <dialog_id>,
    "predictions": [
        {
            "turn_id": <turn_id>,
            "disambiguation_candidates": <bool>,
        }
        ...
    ]
    ...
]

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os
import numpy as np


def compute_precision_recall_f1(n_correct, n_true, n_pred):
    """Computes the precision, recall, and F1 scores.

    Args:
        n_correct: Number of correct (overlapping) predictions
        n_true: Number of ground truth items
        n_pred: Number of items predicted by a model

    Returns:
        rec: Recall
        prec: Precision
        f1: F1 score
    """
    rec = n_correct / n_true if n_true != 0 else 0.
    prec = n_correct / n_pred if n_pred != 0 else 0.
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0.
    return rec, prec, f1

def get_image_name(scene_ids, turn_ind):
    """Given scene ids and turn index, get the image name.
    """
    sorted_scene_ids = sorted(
        ((int(key), val) for key, val in scene_ids.items()),
        key=lambda x: x[0],
        reverse=True
    )
    # NOTE: Hardcoded to only two scenes.
    if turn_ind >= sorted_scene_ids[0][0]:
        scene_label = sorted_scene_ids[0][1]
    else:
        scene_label = sorted_scene_ids[1][1]
    image_label = scene_label
    if "m_" in scene_label:
        image_label = image_label.replace("m_", "")
    # print(scene_label)
    return f"{image_label}.png", scene_label


def get_object_mapping(scene_label):
    """Get the object mapping for a given scene.
    """
    scene_json_path = os.path.join(
        "/home/holy/datasets/simmc2.1/public/", f"{scene_label}_scene.json"
    )
    if not os.path.isfile(scene_json_path):
        scene_json_path = scene_json_path.replace("m_", "")
    with open(scene_json_path, "r") as file_id:
        scene_objects = json.load(file_id)["scenes"][0]["objects"]
    object_map = [ii["index"] for ii in scene_objects]
    return object_map

def evaluate_ambiguous_candidates(
    gt_labels, model_results, record_instance_results=None, is_actually_coref=False,
):
    """Evaluates ambiguous candidates identification subtask.

    Uses golden labels and model predictions.

    Args:
        gt_labels: Ground truth labels.
        model_results: Generated labels.
        record_instance_results: Path to save instance-level metrics.
    """
    gt_label_pool = {ii["dialogue_idx"]: ii for ii in gt_labels["dialogue_data"]}

    predictions = []
    num_evaluations = 0
    num_target_candidates = 0
    num_pred_candidates = 0
    num_overlap_candidates = 0
    for model_datum in model_results:
        dialog_id = model_datum["dialog_id"]
        for round_datum in model_datum["predictions"]:
            round_id = round_datum["turn_id"]
            pred_set = set(round_datum["disambiguation_candidates"])
            gt_datum = gt_label_pool[dialog_id]["dialogue"][round_id]
            # print("gt", gt_datum)
            # print("pred", round_datum)
            # quit()

            # assert "disambiguation_label" in gt_datum["transcript_annotated"], (
            #     "Turn not to be evaluated!"
            # )
            num_evaluations += 1
            if is_actually_coref:
                target_set = set(
                    gt_datum["transcript_annotated"]["act_attributes"]["objects"]
                )
                print("WARNING!!!!\nSome of the coref's target_sets do not exist in the object_map.\nFurther bugfix is needed to correctly evaluate coref!")
            else:
                target_set = set(
                    gt_datum["transcript_annotated"]["disambiguation_candidates"]
                )
            """
            WARNING!!!!
            Some of the coref's target_sets do not exist in the object_map.
            Further bugfix is needed to correctly evaluate coref!
            """
            if target_set != set() and pred_set.intersection(target_set) == set():
                image_name, scene_label = get_image_name(
                    gt_label_pool[dialog_id]["scene_ids"], round_id
                )
                # If dialog contains multiple scenes, map it accordingly.
                object_map = get_object_mapping(scene_label)
                if not target_set.issubset(set(object_map)):
                    print("before", object_map, scene_label)
                    if "m_" in scene_label:
                        object_map = get_object_mapping(f"m_{scene_label}")
                    else:
                        object_map = get_object_mapping(scene_label.replace("m_", ""))
                # print(
                #     "dialog_id", dialog_id,
                #     "| turn_id", round_id,
                #     "| target", target_set,
                #     "| pred", pred_set,
                #     "| intersection", pred_set.intersection(target_set),
                #     "| object_map", object_map,
                #     "| pred_labels", round_datum["disambiguation_labels"])
                # print()
            num_target_candidates += len(target_set)
            num_pred_candidates += len(pred_set)
            num_overlap_candidates += len(pred_set.intersection(target_set))

            # Add the result to datum and save it back.
            if record_instance_results:
                round_datum["ambiguous_candidate_report"] = {
                    "num_pred": len(pred_set),
                    "num_target": len(target_set),
                    "num_overlap": len(pred_set.intersection(target_set)),
                }

    print(f"# Instances evaluated: {num_evaluations}")
    # Record and save per instance results.
    if record_instance_results:
        print("Saving per instance result: {}".format(record_instance_results))
        with open(record_instance_results, "w") as file_id:
            json.dump(model_results, file_id)
    recall, precision, f1 = compute_precision_recall_f1(
        num_overlap_candidates, num_target_candidates, num_pred_candidates
    )
    return {"recall": recall, "precision": precision, "f1": f1}


def main(args):
    print("Reading: {}".format(args["data_json_path"]))
    with open(args["data_json_path"], "r") as file_id:
        gt_labels = json.load(file_id)
    print("Reading: {}".format(args["model_result_path"]))
    with open(args["model_result_path"], "r") as file_id:
        model_results = json.load(file_id)

    if args["record_instance_results"]:
        instance_results_path = args["model_result_path"].replace(
            ".json", "_results.json"
        )
    else:
        instance_results_path = None

    report = evaluate_ambiguous_candidates(
        gt_labels, model_results, record_instance_results=instance_results_path
    )
    print(
        f"""Rec: {report["recall"]:.4f}  |  """
        f"""Prec: {report["precision"]:.4f}  |  """
        f"""F1: {report["f1"]:.4f}"""
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disambiguation Evaluation")
    parser.add_argument(
        "--data_json_path",
        default="data/simmc2.1_dials_dstc11_devtest.json",
        help="Data with gold label for disambiguation",
    )
    parser.add_argument(
        "--model_result_path",
        default=None,
        help="Disambiguation labels generated by the model",
    )
    parser.add_argument(
        "--record_instance_results",
        dest="record_instance_results",
        action="store_true",
        default=False,
        help="Records per instance results and save it back",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
