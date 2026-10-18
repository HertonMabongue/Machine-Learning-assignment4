"""Classification metrics written from the confusion matrix."""
import numpy as np

N_CLASSES = 10


def confusion(y, pred, k=N_CLASSES):
    return np.bincount(y * k + pred, minlength=k * k).reshape(k, k)


def per_class(y, pred, k=N_CLASSES):
    cm = confusion(y, pred, k)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(axis=0), out=np.zeros(k), where=cm.sum(axis=0) > 0)
    rec = np.divide(tp, cm.sum(axis=1), out=np.zeros(k), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return prec, rec, f1, cm.sum(axis=1)


def macro_f1(y, pred):
    return per_class(y, pred)[2].mean()


def weighted_f1(y, pred):
    _, _, f1, sup = per_class(y, pred)
    return float((f1 * sup).sum() / sup.sum())


def accuracy(y, pred):
    return float((y == pred).mean())


def summary(y, pred):
    return dict(accuracy=accuracy(y, pred), macro_f1=macro_f1(y, pred), weighted_f1=weighted_f1(y, pred))
