



## Traffic classification



### Requirements:

For OmniScaleCNN, gMLP, PatchTST, and Sequencer, we use the off-the-shell implentation by the `tsai` package. Here is the requirements:

```python
tsai==0.3.9
```

For others (i.e., DT, TS2Vec, and FormerTime), the requirements (in a new python env) are:

```python
Bottleneck==1.3.5
einops==0.8.0
matplotlib==3.5.3
mkl_service==2.4.0
numpy==1.23.0
pandas==1.4.4
scikit_learn==1.4.2
scipy==1.9.1
statsmodels==0.14.2
torch==1.9.1
tqdm==4.64.1
```



### Data processing:

Run `data/data_ana.ipynb`, generating the `traffic_data.pkl` as the paired samples for model training and testing.

#### load data set:

```python
  import pickle
  data = pickle.load(open('./data/traffic_data.pkl', 'rb'))  
  train_x = data['x'][0].reshape(-1, 3, 24)[:, :].transpose(0, 2, 1) # train data input
  test_y = data['x'][1].reshape(-1, 3, 24)[:, :].transpose(0, 2, 1) # test data input
  train_labels = data['y'][0] # train data label
  test_labels = data['y'][1] # test data label
```



### Usage:

```bash
### for TS2Vec
python ./methods/ts2vec/train.py

### for FormerTime
python ./methods/FormerTime/main.py

### for OmniScaleCNN, gMLP, and Sequencer
bash ./methods/run_OgT.sh

### for PatchTST
python ./methods/train_PatchTST.py
```



### Eval:

Run `eval.ipynb`




