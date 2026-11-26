import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score


def sentiment_metrics(truth, prediction, dataset='sims'):
    truth, prediction = np.asarray(truth).reshape(-1), np.asarray(prediction).reshape(-1)
    if not np.issubdtype(truth.dtype,np.floating):truth=truth.astype(np.float64)
    if not np.issubdtype(prediction.dtype,np.floating):prediction=prediction.astype(np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions')
    if dataset in ('mosi', 'mosei'):
        nonzero = truth != 0
        return {'n': len(truth),
                'acc2': float(accuracy_score(truth >= 0, prediction >= 0)),
                'f1_weighted': float(f1_score(truth >= 0, prediction >= 0, average='weighted', zero_division=0)),
                'acc2_nonzero': float(accuracy_score(truth[nonzero] > 0, prediction[nonzero] > 0)),
                'f1_nonzero': float(f1_score(truth[nonzero] > 0, prediction[nonzero] > 0, average='weighted', zero_division=0)),
                'acc7': float(accuracy_score(np.round(np.clip(truth, -3, 3)), np.round(np.clip(prediction, -3, 3)))),
                'mae': float(np.abs(truth-prediction).mean()),
                'corr': float(np.corrcoef(truth,prediction)[0,1]) if np.std(truth)>0 and np.std(prediction)>0 else 0.0}
    t, p = np.clip(truth, -1, 1), np.clip(prediction, -1, 1)
    t3 = np.digitize(t, np.asarray([-.1,.1],dtype=t.dtype), right=True)
    p3 = np.digitize(p, np.asarray([-.1,.1],dtype=p.dtype), right=True)
    return {'n': len(t), 'acc2': float(accuracy_score(t > 0, p > 0)),
            'f1_weighted': float(f1_score(t > 0, p > 0, average='weighted', zero_division=0)),
            'acc3': float(accuracy_score(t3, p3)),
            'f1_macro3': float(f1_score(t3, p3, labels=[0,1,2], average='macro', zero_division=0)),
            'acc5': float(accuracy_score(np.digitize(t, np.asarray([-.7,-.1,.1,.7],dtype=t.dtype), right=True), np.digitize(p, np.asarray([-.7,-.1,.1,.7],dtype=p.dtype), right=True))),
            'mae': float(np.abs(t-p).mean()),
            'corr': float(np.corrcoef(t,p)[0,1]) if np.std(t)>0 and np.std(p)>0 else 0.0,
            'confusion3': confusion_matrix(t3,p3,labels=[0,1,2]).tolist()}


def conflict_metrics(truth, prediction, deadband=.1):
    t, a, v = [np.asarray(truth[m]).reshape(-1) for m in 'TAV']
    target = ((t*a) < 0) | ((t*v) < 0)
    pt, pa, pv = [np.asarray(prediction[m]).reshape(-1) for m in 'TAV']
    score = np.maximum(-pt*pa, -pt*pv)
    pt,pa,pv = [np.where(np.abs(x)<=deadband,0,x) for x in [pt,pa,pv]]
    pred = ((pt*pa)<0) | ((pt*pv)<0)
    return {'acc2': float(accuracy_score(target,pred)),
            'f1_conflict': float(f1_score(target,pred,zero_division=0)),
            'precision': float(precision_score(target,pred,zero_division=0)),
            'recall': float(recall_score(target,pred,zero_division=0)),
            'average_precision_raw_product': float(average_precision_score(target,score)) if target.any() else 0.0,
            'confusion_tn_fp_fn_tp': confusion_matrix(target,pred,labels=[False,True]).ravel().tolist(),
            'prediction_deadband':deadband, 'target':'T*A<0 OR T*V<0; exact-zero is neutral'}
