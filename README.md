## Acknowledgment 
This project builds upon Sampanna Yashwant Kahu's project, found at: [repository](https://github.com/SampannaKahu/KernelOracle), [paper](https://arxiv.org/abs/2505.15213).

## Abstract
Process scheduling is one of the most performance-critical
components of an operating system. The Linux Completely
Fair Scheduler (CFS) makes decisions purely from instantaneous
kernel state, with no memory of historical execution
patterns. KernelOracle demonstrated that scheduling sequences
are learnable from traces using an LSTM network, but
the model was trained on only a single workload and inference
latency was far too high for real-time use. The goal of this
project is to address both, building off of this repository mentioned in the acknowledgement.
We build a new data collection pipeline using Linux ftrace
to capture scheduling traces across diverse workloads, including 
CPU-bound, I/O-mixed, and scheduler-stress scenarios.
We then replace KernalOracle’s Long Short Term Memory
(LSTM) approach with a Temporal Convolutional Network
(TCN), an architecture that processes sequences in parallel
and maintains lower inference latency. After five epochs of
training, our TCN achieves accuracy and inference latency
which outperforms the LSTM baseline. Our results show that
Linux scheduling behavior is highly predictable across diverse
workloads, and that a TCN is a more suitable model architecture
than an LSTM for this task. 

Our full paper can be found [here](https://github.com/mmarcotullio/KernelOracle/blob/main/report.pdf). 


## Setup

**1. Clone the repo**
```bash
git clone https://github.com/mmarcotullio/KernelOracle
cd KernelOracle
```

**2. Create Python virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Scheduling Trace Data Collection

**Optional, if you'd like to pull the repo's existing trace data files:**
```bash
git lfs install
git lfs pull
```

**Optional, if you'd like to collect your own trace data (requires root):** 
```bash
cd data
sudo python3 collect_traces.py all
python3 split_by_workload.py traces/test_seen.csv
```
This builds test_seen.csv, test_unseen.csv, train.csv, and the split by workloads test CSVs in /data/traces.

## Code Breakdown
The code in the `data` directory contains python scripts that are utilized for the data generation, which include creating the dataset, splitting into train and test set and by workload, and generating a test set for unseen workloads. The commands above walk you through the data generation process.

The `lstm` and `tcn` directories contain the code we used for the model architectures, the data preprocessing, and metrics helpers such as top-1 accuracy. These commands are shown in the respective jupyter notebooks that will be explained in the next section of this guide.

## Setting Up Google Drive and Training The Models
The files `lstm_training_and_evaluation.ipynb` and `tcn_training_and_evaluation.ipynb` are the ones that are being used for model trainings for the LSTM and TCN respectively. They contain the code and commands that should be run in order to go through the data preprocessing, training, and evaluation. 

In order to train the models, the best way to train is by downloading the python notebook files into your Google Drive account. The main reason is that these notebooks are designed to run on Google Colab due to its ability to train with GPUs, and we used the fastest GPU, the A100. Additionally, due to potential constraints that Git LFS may have in terms of quota limits, I would download the csv files under `data/traces` (or the ones that you generated) and create a folder in Google Drive called `trace_data` and load those csv files into the `trace_data` folder. This path is very important because it will be used when you have to mount the drive while running the python notebooks.

Once you download the necessary files into the google drive, the workflow of the notebooks does the following:
1. Cloning the repository
2. Mounting to your google drive for file path compatibility
3. Running the model's data preprocessing script (code is under `<model_type>/scripts/preprocess.py`) for the training, seen test, and unseen test sets. This script converts the csv files into npz files. (NOTE: In the TCN notebook, we also ran this script for the workload specific csv files to obtain accuracy comparisons among each workload).
4. Run a command that trains the model and gets the top-1 accuracy (code is under `<model_type>/train_<model_type>.py`). 
5. Force switch the device from GPU to CPU to get efficiency metrics, including inference latency.

Disclaimer: Depending on your machine, the actual values for the inference latency and throughput might vary depending on the user because it is run on the cpu and each machine is different. We still expect that the trends will stay mostly consistent. We also expect that top-1 accuracy is likely to be similar values for anyone executing these notebooks. 
