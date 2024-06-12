from tsai.basics import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='PatchTST')
args = parser.parse_args()

if 'sp' in args.modelname:
    select_vars = [0]
if 'occ' in args.modelname:
    select_vars = [1]
if 'flow' in args.modelname:
    select_vars = [2]
else:
    select_vars = [0, 1, 2]

import pickle as pkl
data = pkl.load(open('../data/traffic_data.pkl', 'rb'))
if len(select_vars) > 1:
    X, y = np.concatenate([data['x'][0].reshape(-1, 1, 72), data['x'][1].reshape(-1, 1, 72)]), np.concatenate([data['y'][0], data['y'][1]])
else:
    X, y = np.concatenate([data['x'][0].reshape(-1, 1, 24), data['x'][1].reshape(-1, 1, 24)]), np.concatenate([data['y'][0], data['y'][1]])
splits = [np.arange(data['x'][0].shape[0]).tolist(), np.arange(data['x'][0].shape[0], X.shape[0]).tolist()]

### train
tfms = [None, TSClassification()]
batch_tfms = TSStandardize()
clf = TSClassifier(X, y, splits=splits, path='models', arch=args.modelname.split("_")[0], tfms=tfms, batch_size=128, metrics=accuracy)
clf.fit_one_cycle(100, 3e-4)
clf.export(f"{args.modelname}.pkl") 

### testing
mode_name = args.modelname.split("_")[0]
clf = load_learner(f"../ckpts/{mode_name}.pkl")
probas, target, preds = clf.get_X_preds(X[splits[1]], y[splits[1]])
np.save(f"../results/{mode_name}_preds.npy", preds)
print(target.shape, preds.shape)



