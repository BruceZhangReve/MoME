import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe


def MSE_dim(pred, true):
    return np.mean((pred-true)**2, axis=(0,1))


def metric_mixed(pred, true, cat_indices=[-2, -1], vocab_sizes=[3, 3]):
    """
    pred: [bs, out_len, n_vars]
    true: [bs, out_len, n_vars]
    """
    bs, out_len, n_vars = pred.shape

    pred_num = pred[:,:,:n_vars-len(cat_indices)]
    pred_cat = pred[:,:,n_vars-len(cat_indices):].astype(np.int64)#.long()
    #print("shape: ",pred_cat.shape)
    true_num = true[:,:,:n_vars-len(cat_indices)]
    true_cat = true[:,:,n_vars-len(cat_indices):].astype(np.int64)#.long()

    # Numerical metrics
    mae = MAE(pred_num, true_num)
    mse = MSE(pred_num, true_num)
    rmse = RMSE(pred_num, true_num)
    mape = MAPE(pred_num, true_num)
    mspe = MSPE(pred_num, true_num)

    # Classification metrics (Accuracy + F1)
    acc_list = []
    f1_list = []
    for i, vocab_size in enumerate(vocab_sizes):
        pred_labels = pred_cat[:,i,:]         # [bs, out_len]
        true_labels = true_cat[:,i,:]         # [bs, out_len, cat_num]

        pred_flat = pred_labels.reshape(-1)#.cpu().numpy()
        true_flat = true_labels.reshape(-1)#.cpu().numpy() #[]

        #roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")

        acc = accuracy_score(true_flat, pred_flat)
        f1 = f1_score(true_flat, pred_flat, average='micro', zero_division=0)

        acc_list.append(acc)
        f1_list.append(f1)


    return mae, mse, rmse, mape, mspe, acc_list, f1_list

