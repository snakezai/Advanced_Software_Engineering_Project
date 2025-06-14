import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy
import numpy as np
from pykt.models.evaluate_model import evaluate
from torch.autograd import Variable, grad
from pykt.models.atkt import _l2_normalize_adv
from pykt.utils import debug_print
from pykt.config import que_type_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    model_name = model.model_name

    if model_name in ["dkt", "dkt_forget", "dkvmn", "deep_irt", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt",
                      "skvmn", "hawkes"]:

        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # loss = binary_cross_entropy(y.double(), t.double())
        t_ones = torch.ones_like(t)
        loss = binary_cross_entropy(y.double(), t_ones.double())
    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y_next.double(), r_next.double())

        loss_r = binary_cross_entropy(y_curr.double(),
                                      r_curr.double())  # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
    elif model_name in ["akt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos",
                        "akt_raschx", "akt_raschy", "aktvec_raschx"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        criterion = nn.BCELoss(reduction='none')
        loss = criterion(y, t).sum()

    return loss


def model_forward(model, data):
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    if model_name in ["dkt_forget"]:
        dcur, dgaps = data
    else:
        dcur = data
    q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]
    qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
    m, sm = dcur["masks"], dcur["smasks"]

    ys, preloss = [], []
    cq = torch.cat((q[:, 0:1], qshft), dim=1)
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:, 0:1], tshft), dim=1)
    if model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:, 0:1], dcur["shft_itseqs"]), dim=1)
    if model_name in ["dkt"]:
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)  # first: yshft
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn", "deep_irt", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:, 1:])
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name in ["akt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos",
                        "akt_raschx", "akt_raschy", "aktvec_raschx"]:
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:, 1:])
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm)
        # at
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        pred_res, _ = model(c.long(), r.long(), p_adv)
        # second loss
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm)
        loss = loss + model.beta * adv_loss
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)
        # cal loss
    elif model_name == "lpkt":
        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])
    elif model_name == "hawkes":
        # ct = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        # csm = torch.cat((dcur["smasks"][:,0:1], dcur["smasks"]), dim=1)
        # y = model(cc[0:1,0:5].long(), cq[0:1,0:5].long(), ct[0:1,0:5].long(), cr[0:1,0:5].long(), csm[0:1,0:5].long())
        y = model(cc.long(), cq.long(), ct.long(), cr.long())  # , csm.long())
        ys.append(y[:, 1:])
    elif model_name in que_type_models:
        y, loss = model.train_one_step(data)

    if model_name not in ["atkt", "atktfix"] + que_type_models:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    return loss


def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None,
                test_window_loader=None, save_model=False):
    max_auc, best_epoch = 0, -1
    train_step = 0
    if model.model_name == 'lpkt':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step += 1
            if model.model_name in que_type_models:
                model.model.train()
            else:
                model.train()
            loss = model_forward(model, data)
            opt.zero_grad()
            loss.backward()  # compute gradients
            opt.step()  # update model’s parameters

            loss_mean.append(loss.detach().cpu().numpy())
            if model.model_name == "gkt" and train_step % 10 == 0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text=text, fuc_name="train_model")
        if model.model_name == 'lpkt':
            scheduler.step()  # update each epoch
        loss_mean = np.mean(loss_mean)

        auc, acc = evaluate(model, valid_loader, model.model_name)
        ### atkt 有diff， 以下代码导致的
        ### auc, acc = round(auc, 4), round(acc, 4)

        if auc > max_auc + 1e-3:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type + "_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type + "_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name,
                                                              save_test_path)
            validauc, validacc = auc, acc
        print(
            f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(
            f"            testauc: {round(testauc, 4)}, testacc: {round(testacc, 4)}, window_testauc: {round(window_testauc, 4)}, window_testacc: {round(window_testacc, 4)}")

        if i - best_epoch >= 10:
            break
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch