import pickle
import pandas as pd
import os
import torch
import numpy as np


def get_cidx(df):
    """add global id for each interaction"""
    get_cidx = []
    bias = 0
    inter_num = 0
    for _, row in df.iterrows():
        line = row['responses'].split(',')
        cnt = 0
        for i in range(len(line) - 1, -1, -1):
            if line[i] != '-1':
                break
            cnt += 1
        ids_list = [str(x + bias)
                    for x in range(len(row['responses'].split(',')) - cnt)]
        inter_num += len(ids_list)
        ids = ",".join(ids_list)
        get_cidx.append(ids)
        bias += len(ids_list)
    assert inter_num - 1 == int(ids_list[-1])

    return get_cidx


def string_to_list(string, sep=","):
    return [int(x) for x in string.split(sep)]


# LP4Rec处理过的数据是已经提取过最后一个习题id的，需要把提取的习题插入回去
def insert_lastitem(row):
    data_line_map = {column_name: string_to_list(row[column_name]) for column_name in
                     ["questions", "concepts", "responses", "timestamps", "selectmasks", "is_repeat"]}
    last_item_id = df_extracted["last_question"][row.name]


    concepts_indert = concept_map[last_item_id].split(',')[0]

    # 插入的习题是用于kt模型来得到结果的，LP4Rec的实验指标并不需要关注这个，因此在此的最后一个习题的回应设0
    responses_insert = 0

    # LP4Rec处理过的数据是以200分隔的序列抽最后一个去测试的，因此需要插回去进行回溯
    if -1 in data_line_map['questions']:
        insert_idx = data_line_map['questions'].index(-1)
        row['questions'] = data_line_map['questions'][:insert_idx] + [last_item_id] + data_line_map['questions'][insert_idx:]

        row['concepts'] = data_line_map['concepts'][:insert_idx] + [concepts_indert] + data_line_map['concepts'][insert_idx:]
        row['responses'] = data_line_map['responses'][:insert_idx] + [responses_insert] + data_line_map['responses'][insert_idx:]

        row['timestamps'] = data_line_map['timestamps'][:insert_idx] + [max(data_line_map['timestamps'])] + data_line_map['timestamps'][insert_idx:]
        row['selectmasks'] = [1] * insert_idx + [1] + data_line_map['selectmasks'][insert_idx:]
        row['is_repeat'] = data_line_map['is_repeat'][:insert_idx] + [0] + data_line_map['is_repeat'][insert_idx:]
    else:
        row['questions'] = data_line_map['questions'] + [last_item_id]

        row['concepts'] = data_line_map['concepts'] + [concepts_indert]
        row['responses'] = data_line_map['responses'] + [responses_insert]

        row['timestamps'] = data_line_map['timestamps'] + [max(data_line_map['timestamps'])]
        row['selectmasks'] = [1] * (len(data_line_map['questions']) + 1)
        row['is_repeat'] = data_line_map['is_repeat'] + [0]

    for column_name in ["questions", "concepts", "responses", "timestamps", "selectmasks", "is_repeat"]:
        row[column_name] = [str(item) for item in row[column_name]]
        row[column_name] = ','.join(row[column_name])
    return row


def pad_to_length(lst, target_length, pad_value=-1):
    return lst + [pad_value] * (target_length - len(lst)) if len(lst) < target_length else lst[:target_length]

def aligin_testsequences(row):
    concept_ids = list(map(int, row['concepts'].split(',')))
    question_ids = list(map(int, row['questions'].split(',')))
    response_ids = list(map(int, row['responses'].split(',')))
    selectmask_ids = list(map(int, row['selectmasks'].split(',')))
    is_repeat_ids = list(map(int, row['is_repeat'].split(',')))
    timestamps_ids = list(map(int, row['timestamps'].split(',')))

    for i in range(len(question_ids)):
        if question_ids[i] == -1:
            concept_ids[i] = -1
            response_ids[i] = -1
            selectmask_ids[i] = -1
            is_repeat_ids[i] = -1
            timestamps_ids[i] = -1

    row['concepts'] = ','.join(map(str, concept_ids))
    row['responses'] = ','.join(map(str, response_ids))
    row['selectmasks'] = ','.join(map(str, selectmask_ids))
    row['is_repeat'] = ','.join(map(str, is_repeat_ids))
    row['timestamps'] = ','.join(map(str, timestamps_ids))

    return row

def padding_to_maxlen(data_line_map, maxlen):
    for key in data_line_map:
        cur_len = len(data_line_map[key])
        if cur_len < maxlen:
            padding_cnt = maxlen - cur_len
            data_line_map[key].extend([-1] * padding_cnt)

def insert_topk_k(row, k, topk_data, stu_ks, maxlen):
    columns_names = ["questions", "concepts", "responses", "timestamps", "selectmasks", "is_repeat"]
    data_line_map = {column_name: string_to_list(row[column_name]) for column_name in columns_names}
    topk_k = topk_data.iloc[row.name][k]

    # 存在极少推荐-1的情况需要做一步判断否则字典会报错
    if topk_k != -1:
        concepts_insert = concept_map[topk_k].split(',')
        concepts_insert = [int(concept) for concept in concepts_insert]

        # responses_insert = 1 if stu_ks[row.name][concepts_indert] >= 0.5 else 0
        responses_insert = [1 if stu_ks[row.name][concept] >= 0.5 else 0 for concept in concepts_insert]

        # 假如有-1插入到最前面的-1前，没有则插入到最后一个数前
        insert_idx = data_line_map['questions'].index(-1) - 1 if -1 in data_line_map['questions'] else -1

        for i, concept in enumerate(concepts_insert):
            data_line_map['questions'].insert(insert_idx, topk_k)
            data_line_map['concepts'].insert(insert_idx, concept)
            data_line_map['responses'].insert(insert_idx, responses_insert[i])
            data_line_map['timestamps'].insert(insert_idx, max(data_line_map['timestamps']))
            data_line_map['selectmasks'].insert(insert_idx, 1)
            data_line_map['is_repeat'].insert(insert_idx, 0 if i == 0 else 1)

    # 统一填充
    padding_to_maxlen(data_line_map, maxlen)

    row['questions'] = data_line_map['questions']
    row['concepts'] = data_line_map['concepts']
    row['responses'] = data_line_map['responses']
    row['timestamps'] = data_line_map['timestamps']
    row['selectmasks'] = data_line_map['selectmasks']
    row['is_repeat'] = data_line_map['is_repeat']

    for column_name in columns_names:
        row[column_name] = [str(item) for item in row[column_name]]
        row[column_name] = ','.join(row[column_name])
    return row

if __name__ == "__main__":
    # datasetname = ['nips_task34', 'algebra2005', 'assist2017']
    datasetnames = ['nips_task34']
    testmodel = 'GRU4Rec_random'

    # ------------------------------------------------------数据集提取--------------------------------------------------

    # data_path = f'data/{datasetname}/model/{testmodel}/'
    # # 回溯成kt的测试集
    # test_path = data_path + 'data/test_sequences.csv'
    # df = pd.read_csv(test_path)
    #
    # columns_names = ['fold', 'uid', "questions", "concepts", "responses", "timestamps", "selectmasks", "is_repeat", "last_question"]
    #
    # df_extracted = df[columns_names]
    # df_extracted.loc[:, 'questions'] = df['questions'].str.strip(',')
    #
    # concept_map = pd.read_csv(data_path + 'question_concept_map.csv')
    # concept_map = dict(zip(concept_map['question_id'], concept_map['concept_id']))
    # df_extracted = df_extracted.apply(lambda row: insert_lastitem(row), axis=1)
    #
    # columns = ['fold', 'uid', 'questions', 'concepts', 'responses', 'timestamps', 'selectmasks', 'is_repeat']
    # df_extracted = df_extracted[columns]
    #
    # for column in columns[2:]:
    #     df_extracted.loc[:, column] = df_extracted[column].apply(
    #         lambda x: ','.join(map(str, pad_to_length(list(map(int, x.split(','))), 200)))
    #     )
    #
    # df_extracted = df_extracted.apply(aligin_testsequences, axis=1)
    # df_extracted['cidxs'] = get_cidx(df_extracted)
    # df_extracted.to_csv(data_path + 'test_sequences.csv', index=False)


# --------------------------------------------------习题映射转原习题id--------------------------------------------------------

    for datasetname in datasetnames:
        topk_data = pd.read_csv(f'data/{datasetname}/model/{testmodel}/data/topk_data.csv')
        item_id_map = pd.read_csv(f'data/{datasetname}/model/{testmodel}/data/item_token_id.csv')
        token_to_origin = dict(zip(item_id_map['item_token_ID'], item_id_map['origin_item_id']))
        for i in range(1, 11):
            topk_data[f'topk_{i}'] = topk_data[f'topk_{i}'].map(token_to_origin)

        topk_data.to_csv(f'data/{datasetname}/model/{testmodel}/topk_data.csv', index=False)

# --------------------------------------------------插入习题id--------------------------------------------------------

    for datasetname in datasetnames:
        insert_dataset_pth = f'data/{datasetname}/model/{testmodel}/test_sequences.csv'
        topk_data = pd.read_csv(f'data/{datasetname}/model/{testmodel}/topk_data.csv')
        concept_map = pd.read_csv(f'data/{datasetname}/model/{testmodel}/question_concept_map.csv')

        stu_ks_pth = f'data/{datasetname}/model/{testmodel}/ori_pkc.pth'
        stu_ks = torch.load(stu_ks_pth)
        # 选择插入topk进数据集
        topk_to_insert = 10


        concept_map = dict(zip(concept_map['question_id'], concept_map['concept_id']))
        maxlen = max(len(l.split(',')) for l in concept_map.values()) + 200  # 200是原本测试集划分的长度,这里统一测试集一致的长度
        stu_ks = stu_ks.tolist()

        for k in range(1, topk_to_insert + 1):
            df_dataset = pd.read_csv(insert_dataset_pth)
            df_dataset = df_dataset.apply(lambda row: insert_topk_k(row, k, topk_data, stu_ks, maxlen), axis=1)
            df_dataset['cidxs'] = get_cidx(df_dataset)
            os.makedirs(f'data/{datasetname}/model/{testmodel}/Topk_insert_data/', exist_ok=True)
            df_dataset.to_csv(f'data/{datasetname}/model/{testmodel}/Topk_insert_data/top{k}_insert_testdata.csv', index=False)

    # --------------------------------------------------------对他齐其他插入数据----------------------------------------------------------

    # datasetname = 'nips_task34'
    # testmodel = 'mmer'
    # for i in range(1, 11):
    #     df = pd.read_csv(f'data/{datasetname}/model/{testmodel}/Topk_insert_data/top{i}_insert_testdata.csv')
    #     # df = df.apply(aligin_testsequences, axis=1)
    #     df['cidxs'] = get_cidx(df)
    #     df.to_csv(f'data/{datasetname}/model/{testmodel}/Topk_insert_data/top{i}_insert_testdata.csv', index=False)