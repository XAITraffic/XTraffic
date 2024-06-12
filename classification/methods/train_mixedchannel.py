from tsai.basics import *

import argparse
import os
import sys
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='OmniScaleCNN')
args = parser.parse_args()

if 'sp' in args.modelname:
    select_vars = [0]
if 'occ' in args.modelname:
    select_vars = [1]
if 'flow' in args.modelname:
    select_vars = [2]
else:
    select_vars = [0, 1, 2]

data = pkl.load(open('../traffic_data.pkl', 'rb'))
X, y = np.concatenate([data['x'][0].reshape(-1, 3, 24), data['x'][1].reshape(-1, 3, 24)]), np.concatenate([data['y'][0], data['y'][1]])
splits = [np.arange(data['x'][0].shape[0]).tolist(), np.arange(data['x'][0].shape[0], X.shape[0]).tolist()]

### training
tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
clf = TSClassifier(X, y, splits=splits, path='models', arch=args.modelname.split("_")[0], tfms=tfms, batch_size=128, metrics=accuracy, sel_vars=select_vars)
clf.fit_one_cycle(100, 3e-4)
clf.export(f"../ckpts/{args.modelname}.pkl") 


### testing 
clf = load_learner(f"../ckpts/{args.modelname}.pkl")
probas, target, preds = clf.get_X_preds(X[splits[1]], y[splits[1]])
np.save(f"../results/{args.modelname}_preds.npy", preds)
print(target.shape, preds.shape)