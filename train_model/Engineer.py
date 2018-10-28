# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import sys
import os
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from global_variables.global_variables import use_cuda
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from config.config import cfg
from tools.timer import Timer


def masked_unk_softmax(x, dim, mask_idx):
    x1 = F.softmax(x, dim=dim)
    x1[:, mask_idx] = 0
    x1_sum = torch.sum(x1, dim=1, keepdim=True)
    y = x1 / x1_sum
    return y


def compute_score_with_logits(logits, labels):
    logits = masked_unk_softmax(logits, 1, 0)
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size())
    one_hots = one_hots.cuda() if use_cuda else one_hots
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def clip_gradients(myModel, i_iter, writer):
    max_grad_l2_norm = cfg.training_parameters.max_grad_l2_norm
    clip_norm_mode = cfg.training_parameters.clip_norm_mode
    if max_grad_l2_norm is not None:
        if clip_norm_mode == 'all':
            norm = nn.utils.clip_grad_norm_(myModel.parameters(), max_grad_l2_norm)
            writer.add_scalar('grad_norm', norm, i_iter)
        elif clip_norm_mode == 'question':
            norm = nn.utils.clip_grad_norm_(myModel.module.question_embedding_models.parameters(),
                                            max_grad_l2_norm)
            writer.add_scalar('question_grad_norm', norm, i_iter)
        else:
            raise NotImplementedError


def check_params_and_grads(myModel):
    params_ok = True
    for name, param in myModel.named_parameters():
        if torch.isnan(param).any():
            params_ok = False
            print("NaN detected in param: {}".format(name))

        if param.grad is not None:
            if torch.isnan(param.grad).any():
                params_ok = False
                print("NaN detected in grad: {}".format(name))
            if param.grad.gt(1e6).any():
                params_ok = False
                print("Exploding grad: {}".format(name))
        else:
            print("No grad: {} ({})".format(name, param.grad))
    return params_ok


def save_a_report(i_iter, train_loss, train_acc, train_avg_acc, report_timer, writer, data_reader_eval,
                  my_model, model_type, loss_criterion):
    val_batch = next(iter(data_reader_eval))
    _, val_score, val_loss, n_val_sample = compute_a_batch(val_batch, my_model, run_fn=get_run_fn(model_type), eval_mode=True, loss_criterion=loss_criterion)
    val_acc = val_score / n_val_sample

    print("iter:", i_iter, "time(s): % s \n" % report_timer.end(),
          "Model: {}".format(model_type),
          "train_loss: %.4f" % train_loss,
          "train_score: %.4f" % train_acc,
          "avg_train_score: %.4f" % train_avg_acc,
          "val_score: %.4f" % val_acc,
          "val_loss: %.4f" % val_loss.item()
          )
    sys.stdout.flush()
    report_timer.start()

    writer.add_scalar('loss/train', train_loss, i_iter)
    writer.add_scalar('score/train', train_acc, i_iter)
    writer.add_scalar('score/avg_train', train_avg_acc, i_iter)
    writer.add_scalar('score/val', val_acc, i_iter)
    writer.add_scalar('loss/val', val_loss.item(), i_iter)
    for name, param in my_model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), i_iter)


def save_a_snapshot(snapshot_dir,i_iter, iepoch, main_model, adv_model,
                    main_optimizer, adv_optimizer,
                    loss_criterion, best_val_accuracy, best_epoch, best_iter,
                    snapshot_timer, data_reader_eval, main_train_acc):
    model_snapshot_file = os.path.join(snapshot_dir, "model_%08d.pth" % i_iter)
    model_result_file = os.path.join(snapshot_dir, "result_on_val.txt")
    save_dic = {
        'epoch': iepoch,
        'iter': i_iter,
        'state_dict': main_model.state_dict(),
        'adv_state_dict': adv_model.state_dict(),
        'optimizer': main_optimizer.state_dict(),
        'adv_optimizer': adv_optimizer.state_dict()}

    if data_reader_eval is not None:
        val_accuracy, avg_loss, val_sample_tot = one_stage_eval_model(data_reader_eval, main_model, get_run_fn('main'),
                                                                      loss_criterion=loss_criterion)
        print("i_epoch:", iepoch, "i_iter:", i_iter, "val_loss:%.4f" % avg_loss,
              "val_acc:%.4f" % val_accuracy, "runtime: %s" % snapshot_timer.end())
        snapshot_timer.start()
        sys.stdout.flush()

        with open(model_result_file, 'a') as fid:
            fid.write('%d %d %.5f %.5f\n' % (iepoch, i_iter, val_accuracy, main_train_acc))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = iepoch
            best_iter = i_iter
            best_model_snapshot_file = os.path.join(snapshot_dir, "best_model.pth")

        save_dic['best_val_accuracy'] = best_val_accuracy
        torch.save(save_dic, model_snapshot_file)

        if best_iter == i_iter:
            if os.path.exists(best_model_snapshot_file):
                os.remove(best_model_snapshot_file)
            os.link(model_snapshot_file, best_model_snapshot_file)

    return best_val_accuracy, best_epoch, best_iter


def one_stage_train(main_model, adv_model, data_reader_trn, main_optimizer, adv_optimizer,
                    loss_criterion, snapshot_dir, log_dir,
                    i_iter, start_epoch, best_val_accuracy=0, data_reader_eval=None,
                    scheduler=None, adv_scheduler=None):
    report_interval = cfg.training_parameters.report_interval
    snapshot_interval = cfg.training_parameters.snapshot_interval

    max_iter = cfg.training_parameters.max_iter

    lambda_q = cfg.training_parameters.lambda_q

    main_avg_accuracy = adv_avg_accuracy = 0
    accuracy_decay = 0.99
    best_epoch = 0
    main_writer = SummaryWriter(os.path.join(log_dir, 'main'))
    adv_writer = SummaryWriter(os.path.join(log_dir, 'adversary'))
    best_iter = i_iter
    iepoch = start_epoch
    snapshot_timer = Timer('m')
    report_timer = Timer('s')

    if hasattr(main_model, 'module'):
        q_emb = main_model.module.question_embedding_models
    else:
        q_emb = main_model.question_embedding_models

    print("MAX ITER: {}".format(max_iter))

    while i_iter < max_iter:
        iepoch += 1

        main_score_epoch = 0
        n_sample_tot = 0
        for i, batch in enumerate(data_reader_trn):
            i_iter += 1
            if i_iter > max_iter:
                break

            scheduler.step(i_iter)
            adv_scheduler.step(i_iter)

            main_writer.add_scalar('learning_rate', scheduler.get_lr()[0], i_iter)
            adv_writer.add_scalar('learning_rate', adv_scheduler.get_lr()[0], i_iter)

            # Run main model
            main_optimizer.zero_grad()
            main_logits, main_scores, main_loss, n_sample = compute_a_batch(batch,
                                                               main_model,
                                                               run_fn=one_stage_run_model,
                                                               eval_mode=False,
                                                               loss_criterion=loss_criterion)

            main_loss.backward()

            main_qnorm = get_grad_norm(q_emb.parameters())
            main_writer.add_scalar('Q_norm', main_qnorm, i_iter)

            main_accuracy = main_scores / n_sample
            main_avg_accuracy += (1 - accuracy_decay) * (main_accuracy - main_avg_accuracy)
            main_score_epoch += main_scores
            n_sample_tot += n_sample

            clip_gradients(main_model, i_iter, main_writer)
            check_params_and_grads(main_model)
            main_optimizer.step()

            # Run adv model
            if lambda_q > 0:
                adv_optimizer.zero_grad()
                adv_logits, adv_scores, adv_loss, n_sample = compute_a_batch(batch,
                                                                 adv_model,
                                                                 run_fn=one_stage_run_adv,
                                                                 eval_mode=False,
                                                                 loss_criterion=loss_criterion)

                adv_accuracy = adv_scores / n_sample
                adv_avg_accuracy += (1 - accuracy_decay) * (adv_accuracy - adv_avg_accuracy)

                adv_loss *= lambda_q
                adv_loss.backward()

                adv_qnorm = get_grad_norm(q_emb.parameters())
                adv_writer.add_scalar('Q_norm', adv_qnorm, i_iter)

                clip_gradients(adv_model, i_iter, adv_writer)
                check_params_and_grads(adv_model)
                adv_optimizer.step()
            else:
                adv_accuracy = 0
                adv_loss = torch.zeros(1)

            # main_entropy = Categorical(logits=main_logits).entropy()
            # adv_entropy = Categorical(logits=adv_logits).entropy()


            if i_iter % report_interval == 0:
                save_a_report(i_iter, main_loss.item(), main_accuracy, main_avg_accuracy, report_timer,
                              main_writer, data_reader_eval, main_model, 'main', loss_criterion)
                save_a_report(i_iter, adv_loss.item(), adv_accuracy, adv_avg_accuracy, report_timer,
                              adv_writer, data_reader_eval, adv_model, 'adv', loss_criterion)

            if i_iter % snapshot_interval == 0 or i_iter == max_iter:
                main_train_acc = main_score_epoch / n_sample_tot
                best_val_accuracy, best_epoch, best_iter = save_a_snapshot(snapshot_dir, i_iter, iepoch, main_model, adv_model,
                                                                           main_optimizer, adv_optimizer, loss_criterion, best_val_accuracy,
                                                                           best_epoch, best_iter, snapshot_timer,
                                                                           data_reader_eval, main_train_acc)

    main_writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    main_writer.close()
    adv_writer.close()
    print("best_acc:%.6f after epoch: %d/%d at iter %d" % (best_val_accuracy, best_epoch, iepoch, best_iter))
    sys.stdout.flush()


# def evaluate_a_batch(batch, main_model, loss_criterion):
#     answer_scores = batch['ans_scores']
#     n_sample = answer_scores.size(0)
#
#     input_answers_variable = Variable(answer_scores.type(torch.FloatTensor))
#     if use_cuda:
#         input_answers_variable = input_answers_variable.cuda()
#
#     logit_res = one_stage_run_model(batch, main_model)
#     predicted_scores = torch.sum(compute_score_with_logits(logit_res,
#                                  input_answers_variable.data))
#     total_loss = loss_criterion(logit_res, input_answers_variable)
#
#     return predicted_scores / n_sample, total_loss.item()


def compute_a_batch(batch, main_model, run_fn, eval_mode, loss_criterion=None):

    obs_res = batch['ans_scores']
    obs_res = Variable(obs_res.type(torch.FloatTensor))
    if use_cuda:
        obs_res = obs_res.cuda()

    n_sample = obs_res.size(0)
    logits = run_fn(batch, main_model, eval_mode)
    scores = torch.sum(compute_score_with_logits(logits, obs_res.data))
    loss = None if loss_criterion is None else loss_criterion(logits, obs_res)

    return logits, scores, loss, n_sample


def one_stage_eval_model(data_reader_eval, main_model, run_fn, loss_criterion=None):
    score_tot = 0
    n_sample_tot = 0
    loss_tot = 0
    for batch in tqdm(data_reader_eval):
        _, score, loss, n_sample = compute_a_batch(batch, main_model, run_fn, eval_mode=True, loss_criterion=loss_criterion)
        score_tot += score
        n_sample_tot += n_sample
        if loss is not None:
            loss_tot += loss.item() * n_sample
    return score_tot / n_sample_tot, loss_tot / n_sample_tot, n_sample_tot


def one_stage_run_model(batch, main_model, eval_mode):
    if eval_mode:
        main_model.eval()
    else:
        main_model.train()

    input_text_seqs = batch['input_seq_batch']
    input_images = batch['image_feat_batch']
    input_txt_variable = Variable(input_text_seqs.type(torch.LongTensor))
    image_feat_variable = Variable(input_images)
    if use_cuda:
        input_txt_variable = input_txt_variable.cuda()
        image_feat_variable = image_feat_variable.cuda()

    image_feat_variables = [image_feat_variable]

    image_dim_variable = None
    if 'image_dim' in batch:
        image_dims = batch['image_dim']
        image_dim_variable = Variable(image_dims,
                                      requires_grad=False,
                                      volatile=False)
        if use_cuda:
            image_dim_variable = image_dim_variable.cuda()

    # check if more than 1 image_feat_batch
    i = 1
    image_feat_key = "image_feat_batch_%s"
    while image_feat_key % str(i) in batch:
        tmp_image_variable = Variable(batch[image_feat_key % str(i)])
        if use_cuda:
            tmp_image_variable = tmp_image_variable.cuda()
        image_feat_variables.append(tmp_image_variable)
        i += 1

    main_logits = main_model(input_question_variable=input_txt_variable,
                         image_dim_variable=image_dim_variable,
                         image_feat_variables=image_feat_variables)

    return main_logits


def one_stage_run_adv(batch, adv_model, eval_mode):
    if eval_mode:
        adv_model.eval()
    else:
        adv_model.train()

    input_text_seqs = batch['input_seq_batch']
    input_txt_variable = Variable(input_text_seqs.type(torch.LongTensor))
    if use_cuda:
        input_txt_variable = input_txt_variable.cuda()

    adv_logits = adv_model(input_question_variable=input_txt_variable)

    return adv_logits


def get_run_fn(model_type):
    if model_type == 'main':
        return one_stage_run_model
    elif model_type == 'adv':
        return one_stage_run_adv
    else:
        raise ValueError(model_type)


# Like clip_grad_norm_() but just returns the grad norm without clipping
def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm
