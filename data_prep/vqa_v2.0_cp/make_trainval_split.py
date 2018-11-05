# Since VQA-CP has no train/val split, make our own

import json
import os
import random

VAL_SIZE = 0.10

annotation_file = os.path.join('vqacp_v2_train_annotations.json')
question_file = os.path.join('vqacp_v2_train_questions.json')

with open(annotation_file) as f:
    annotations = json.load(f)
qid2ann_dict = {ann['question_id']: ann for ann in annotations}

with open(question_file) as f:
    questions = json.load(f)
random.shuffle(questions)

assert(len(questions) == len(annotations))

split_idx = int(VAL_SIZE * len(questions))

train_questions = []
train_annotations = []
val_questions = []
val_annotations = []

for i, q in enumerate(questions):
    qid = q['question_id']
    ann = qid2ann_dict[qid]
    if i < split_idx:
        val_questions.append(q)
        val_annotations.append(ann)
    else:
        train_questions.append(q)
        train_annotations.append(ann)

with open('vqacp_v2_trainsplit_questions.json', 'w') as outfile:
    json.dump(train_questions, outfile)
with open('vqacp_v2_trainsplit_annotations.json', 'w') as outfile:
    json.dump(train_annotations, outfile)
with open('vqacp_v2_valsplit_questions.json', 'w') as outfile:
    json.dump(val_questions, outfile)
with open('vqacp_v2_valsplit_annotations.json', 'w') as outfile:
    json.dump(val_annotations, outfile)
