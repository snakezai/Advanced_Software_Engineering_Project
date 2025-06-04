import os
import argparse
import json
import copy
import torch
import pandas as pd

from pykt.models import evaluate_question, load_model
from pykt.datasets import init_test_datasets

from evaluate_model import evaluate

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'

with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)


def main(params, laye_id, testmodel):
    if params['use_wandb'] == 1:
        import wandb
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2']:
            if remove_item in model_config:
                del model_config[remove_item]
        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], \
        trained_params["emb_type"]
        if model_name in ["saint", "sakt", "atdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len

    for i in range(1, 11):
        with open("../configs/data_config.json") as fin:
            curconfig = copy.deepcopy(json.load(fin))
            data_config = curconfig[dataset_name]
            data_config["dataset_name"] = dataset_name

            # 更换kt时这里也要换
            data_config["test_file"] = f'model/{testmodel}/mlpLayer{laye_id}/Topk_insert_data/top{i}_insert_testdata.csv'
            dpath = data_config["dpath"]

            if model_name in ["dkt_forget", "bakt_time"]:
                data_config["num_rgap"] = config["data_config"]["num_rgap"]
                data_config["num_sgap"] = config["data_config"]["num_sgap"]
                data_config["num_pcount"] = config["data_config"]["num_pcount"]
            elif model_name == "lpkt":
                data_config["num_at"] = config["data_config"]["num_at"]
                data_config["num_it"] = config["data_config"]["num_it"]
        if model_name not in ["dimkt"]:
            test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(
                data_config, model_name, batch_size)
        else:
            diff_level = trained_params["difficult_levels"]
            test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(
                data_config, model_name, batch_size, diff_level=diff_level)

        print(
            f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
        print(f"model_config: {model_config}")
        print(f"data_config: {data_config}")

        model = load_model(model_name, model_config, data_config, emb_type, save_dir)
        model = model.to(device)

        save_test_path = os.path.join(save_dir, model.emb_type + "_test_predictions.txt")

        if model.model_name == "rkt":
            dpath = data_config["dpath"]
            dataset_name = dpath.split("/")[-1]
            tmp_folds = set(data_config["folds"]) - {fold}
            folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
            rel = None
            if dataset_name in ["algebra2005", "bridge2algebra2006"]:
                fname = "phi_dict" + folds_str + ".pkl"
                rel = pd.read_pickle(os.path.join(dpath, fname))
            else:
                fname = "phi_array" + folds_str + ".pkl"
                rel = pd.read_pickle(os.path.join(dpath, fname))

        if model.model_name == "rkt":
            testauc, testacc, stu_ks = evaluate(model, test_loader, model_name, rel, save_test_path)
        else:
            testauc, testacc, stu_ks = evaluate(model, test_loader, model_name, 'None', save_test_path)
        print(f"testauc: {testauc}, testacc: {testacc}")

        os.makedirs(os.path.join(dpath, f'model/{testmodel}/mlpLayer{laye_id}'), exist_ok=True)
        torch.save(stu_ks, os.path.join(dpath, f'model/{testmodel}/mlpLayer{laye_id}/top{i}_pkc.pth'))

        torch.cuda.empty_cache()


if __name__ == "__main__":
    testmodel = 'LP4Rec'
    # 测试前先设好kt的名字
    layers = [2]


# ------------------------------------------------------------------------------------------------------------------

    # save_model_path = 'nips_task34_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_980880ca-3ca9-4e74-8255-ebfd4e44da9e'
    save_model_path = 'algebra2005_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_d3be0c25-2bcc-4be8-beae-ee53704aa232'
    # save_model_path = 'assist2017_dkt_qid_saved_model_42_0_0.2_200_0.001_0_1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default=f"saved_model/{save_model_path}")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=0)

    for layer in layers:
        args = parser.parse_args()
        print(args)
        params = vars(args)
        main(params, layer, testmodel)
