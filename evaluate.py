import numpy as np
import torch
import pandas as pd

topK = 10

def evaluate(init_ks_path, data_path, test_model):
    init_ks = (torch.load(init_ks_path)).detach().cpu().numpy()
    stu_num = init_ks.shape[0]
    kc_num = len(init_ks[0])
    Ep_np = np.zeros(shape=(stu_num, topK))

    for i in range(topK):
        ks_top_k = (torch.load(f'{data_path}/top{i + 1}_pkc.pth')).detach().cpu().numpy()
        ks = np.mean((ks_top_k - init_ks) / (1 - init_ks), axis=1)
        Ep_np[:, i] = ks

    Ep = {}
    for i in [1, 3, 5, 10]:
        Ep[f'@{i}'] = 0
        for j in range(stu_num):
            Ep[f'@{i}'] += np.sum(Ep_np[j, :i]) / i
        Ep[f'@{i}'] = round(Ep[f'@{i}'] / stu_num, 5)

    print(f'test_model: {test_model}, Ep: {Ep}\n')
    return {'test_model': test_model, 'Ep': Ep}

if __name__ == '__main__':
    dataset_name = 'algebra2005'
    data_path = f'./data/{dataset_name}/model/'
    test_models = ['dkt']

    all_data = []
    for test_model in test_models:
        init_ks_path = data_path + test_model + '/ori_pkc.pth'
        test_path = data_path + test_model
        row = evaluate(init_ks_path, test_path, test_model)
        all_data.append(row)

    df = pd.DataFrame()
    for i, data in enumerate(all_data):
        for metric, values in data.items():
            if metric != 'test_model':
                for k, v in values.items():
                    df.loc[i, f'{metric}{k}'] = v
            else:
                df.loc[i, 'test_model'] = data['test_model']

    df.to_csv(f'{dataset_name}_output.csv', index=False)