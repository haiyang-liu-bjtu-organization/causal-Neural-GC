import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from synthetic import simulate_var
from sklearn.metrics import roc_curve, roc_auc_score
from models.drmlp_new_alter import load_lstm_weights, train_model_ista, train_model_ista_no, drmlp_f, drmlp_ms, drmlp_ls, drmlp_ds, drmlp_nolstm, drmlp_nolstm_s, drmlp_star, drmlp_nolstm_star
from utils.logger import create_logger

# For GPU acceleration
device = torch.device('cuda')

parser = argparse.ArgumentParser(description='DRC-Net args')
parser.add_argument('-t', '--length', type=int, help='Length of data')
parser.add_argument('-sd', '--seed', type=int, help='Seed of data')
parser.add_argument('--varlag', type=int, default=3, help='The Max Lag of Var data')
parser.add_argument('--varsp', type=int, default=2, help='The sparity of Var data')
parser.add_argument('--vardelay', type=int, default=0, help='The delay of Var data')
args = parser.parse_args()


lam = 1e-3
ridge = 1e-2
lr = 5e-2
model_name = 'drmlpds'
param = 'lam-{:.0e}-ridge-{:.0e}-lr-{:.0e}'.format(lam, ridge, lr)
hard = False
switchper = 200


def Mkdir(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


data_dir = './datasets/var/lag%d-sp%d/time%d/' % (args.varlag, args.varsp, args.length)
data_file = 'var%d-%d-%d-0-%d.npz' % (args.varlag, args.seed, args.varsp, args.length)
logger_dir = './exp_logger/part1/%s/' % (model_name)
logger_file = data_file[:-4] + '_%s_%s.log' % (model_name, param)
logger = create_logger(logger_dir, logger_file)
logger.info('------Begin Training Model------')

ars = []
if os.path.exists(data_dir+data_file):
    data = np.load(data_dir+data_file)
    X_np = data['X']
    beta = data['beta']
    GC = data['GC']
    print("已加载"+data_file)
    X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)

model = drmlp_ds(num_series=X.shape[-1], lag=5).cuda()
print(lam, ridge, lr)
train_loss_list = train_model_ista(
    model, X, lam=lam, lam_ridge=ridge, lr=lr, max_iter=50000,
    check_every=100, update_hard=hard, logfile=logger)

GC_est = model.GC().cpu().data.numpy()
GC_est_soft = model.GC(threshold=False).cpu().data.numpy()
GC_est_lag_total = model.GC(ignore_lag=False, threshold=False).cpu().data.numpy()
Train_loss_list_np = torch.tensor(train_loss_list).detach().cpu().data.numpy()

y_true = GC.flatten()
print(y_true)
y_pred_proba = GC_est_soft.flatten()

# 计算 AUROC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
auc_roc = roc_auc_score(y_true, y_pred_proba)
print("AUROC:", auc_roc)

ars.append(auc_roc)

result_dir = './exp_result/part1/%s/' % (model_name)
Mkdir(result_dir)
result_file = data_file[:-4] + '_%s_%s.npz' % (model_name, param)
np.savez(result_dir + result_file, 
    LossList=Train_loss_list_np, 
    GCTrue=GC, 
    GCEst=GC_est, 
    GCEstSoft=GC_est_soft, 
    GCEstLag=GC_est_lag_total,
    FprList=fpr,
    TprList=tpr,
    AUROC=auc_roc)
print('已完成一次实验')

print(auc_roc)
logger.info('seed{}:{}'.format(args.seed, auc_roc))