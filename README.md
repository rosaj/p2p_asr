# Automatic Speech Recognition (ASR) in peer-to-peer environments

### Environment
Before install required packages and python version `3.8`.
```ssh
pip install -r requirements.txt
```

### Dataset
To run the experiments please download `UserLibri` dataset from [https://www.kaggle.com/datasets/google/userlibri](https://www.kaggle.com/datasets/google/userlibri).
The `UserLibri` dataset should be extracted and placed in `data/userlibri` resulting in the following hierarchy:
```ssh
project_root/
│
├── data/
│   ├── userlibri/
│   │   ├── UserLibri
│   │   │   ├── audio_data
│   │   │   ├── lm_data
│
└── other_directories/
```

The `LJ Speech` dataset will be automatically downloaded when running the code for the first time.  


### Running the code
To run a single (centralized) model on all data, refer to the `Central (single) model` notebook.  
Notebook `Peer-to-peer` contains `code` to run ASR model training in a peer-to-peer environment.

The code will run on Nvidia GPUs or you can try to run it on a CPU by changing visible devices to CPU with `set_visible_devices('CPU')`.  



### Adding custom dataset
To add a new dataset you will need to create a new `clients_data.py` since every dataset has it's own way of loading. You may try reusing existing loaders (copying them) for `UserLibri` or `LJ Speech` datasets that are available in the `data/userlibri` and `data/ljspeech` directories.  
File `clients_data.py` must contain function `load_clients_data` that will return a dictionary in the following format (example contains 3 clients data):
```ssh
{
    "train": [
              ([x1, x2, x3, ....], [y1, y2, y3, ....]),
              ([x11, x12, x23, ....], [y11, y12, y23, ....]),
              ([x21, x22, x23, ....], [y21, y22, y23, ....]), 
              ],
    "val": [([], []) for _ in range(num_clients)],
    "test": [
              ([tx1, tx2, tx3, ....], [ty1, ty2, ty3, ....]),
              ([tx11, tx12, tx23, ....], [ty11, ty12, ty23, ....]),
              ([tx21, tx22, tx23, ....], [ty21, ty22, ty23, ....]), 
              ],
    "dataset_name": ['userlibri'] * num_clients, 
}
```
where `x`/`tx` represents spectogram instance and `y`/`ty` the corresponding tokenized textual representation.  
Please inspect how existing `clients_data.py` are prepared and prepare your dataset accordingly.  

When you prepared your custom `clients_data.py` for your custom dataset, simply change the import in the notebook and the model/agents will use data from the custom dataset.  

 
### Training parameters
The default training parameters such as `batch_size` and `learning_rate` are optimally set to optimize the learning process in both centralized and peer-to-peer environment.
`batch_size` of `8` was set to both reduce GPU memory usage and improve learning. `learning_rate` was set to `1e-4` for experiments.
Please note that the training using the `LJ Speech` dataset in peer-to-peer environment will last 7 days to complete 500 epochs on a **NVIDIA GeForce RTX 2080 Ti** with **10.7GB** of GPU RAM memory.
Training using the `UserLibri` dataset will last 3-4 days to complete 500 epochs of local training per agent.
Training a single model in centralized model will take half as much and the model will converge in about 50 epochs.
  
