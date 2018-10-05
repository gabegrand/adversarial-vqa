# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import h5py
import json
import _pickle as pickle
import timeit
import sys
from tqdm import tqdm
import numpy as np
from train_model.Engineer import one_stage_run_model, masked_unk_softmax, compute_a_batch
from train_model.model_factory import prepare_model


class answer_json:
    def __init__(self):
        self.answers = []

    def add(self, ques_id, ans):
        res = {
            "question_id": ques_id,
            "answer": ans
        }
        self.answers.append(res)


def build_model(config, dataset):
    num_vocab_txt = dataset.vocab_dict.num_vocab
    num_choices = dataset.answer_dict.num_vocab

    num_image_feat = len(config['data']['image_feat_train'][0].split(','))
    my_model = prepare_model(num_vocab_txt, num_choices, **config['model'],
                            num_image_feat=num_image_feat)
    return my_model


def z_saver(current_model, data_reader, hdf5_z, UNK_idx=0):
    softmax_tot = []
    q_id_tot = []

    score_tot = 0
    n_sample_tot = 0

    for batch in tqdm(data_reader):
        verbose_info = batch['verbose_info']
        q_ids = verbose_info['question_id'].cpu().numpy().tolist()
        logit_res, joint_embedding, score, n_sample = compute_a_batch(batch, current_model, eval_mode=True)

        joint_embedding = joint_embedding.data.cpu().numpy().astype(np.float16)
        batch_size = int(joint_embedding.shape[0])

        hdf5_z[n_sample_tot:n_sample_tot+batch_size] = joint_embedding

        score_tot += score
        n_sample_tot += n_sample
        softmax_res = masked_unk_softmax(logit_res, dim=1, mask_idx=UNK_idx)
        softmax_res = softmax_res.data.cpu().numpy().astype(np.float16)
        q_id_tot += q_ids
        softmax_tot.append(softmax_res)

    acc = score_tot / n_sample_tot

    return q_id_tot, acc, n_sample_tot


def print_result(question_ids,
                 soft_max_result,
                 ans_dic,
                 out_file,
                 json_only=True,
                 pkl_res_file=None):
    predicted_answers = np.argmax(soft_max_result, axis=1)

    if not json_only:
        with open(pkl_res_file, 'wb') as writeFile:
            pickle.dump(soft_max_result, writeFile)
            pickle.dump(question_ids, writeFile)
            pickle.dump(ans_dic, writeFile)

    ans_json_out = answer_json()
    for idx, pred_idx in enumerate(predicted_answers):
        question_id = question_ids[idx]
        pred_ans = ans_dic.idx2word(pred_idx)
        ans_json_out.add(question_id, pred_ans)

    with open(out_file, "w") as f:
        json.dump(ans_json_out.answers, f)
