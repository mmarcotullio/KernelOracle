## Acknowledgment 
This project builds upon Sampanna Yashwant Kahu's project, found at: [Repository]([paper](https://arxiv.org/abs/2505.15213)), [paper](https://arxiv.org/abs/2505.15213)

## Abstract
Process scheduling is one of the most performance-critical
components of an operating system. The Linux Completely
Fair Scheduler (CFS) makes decisions purely from instanta-
neous kernel state, with no memory of historical execution
patterns. KernelOracle [ 5 ] demonstrated that scheduling se-
quences are learnable from traces using an LSTM network, but
the model was trained on only a single workload and inference
latency was far too high for real-time use. The goal of this
project is to address both, building off of this repository [4 ].
We build a new data collection pipeline using Linux ftrace
to capture scheduling traces across diverse workloads, includ-
ing CPU-bound, I/O-mixed, and scheduler-stress scenarios.
We then replace KernalOracle’s Long Short Term Memory
(LSTM) approach with a Temporal Convolutional Network
(TCN), an architecture that processes sequences in parallel
and maintains lower inference latency. After five epoch of
training, our TCN achieves accuracy and inference latency
which outperforms the LSTM baseline. Our results show that
Linux scheduling behavior is highly predictable across diverse
workloads, and that a TCN is a more suitable model architec-
ture than an LSTM for this task. Our implementation is in this


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


**Optional, if you'd ike to collect your own trace data (requires root). Note: this is not necessary if you pulled the existing trace data files:** 
```bash
cd data
sudo python3 collect_traces.py all
python3 split_by_workload.py traces/test_seen.csv
```
###### this builds test_seen.csv, test_unseen.csv, train.csv, and the split by workloads test CSVs in /data/traces




## Below is the baseline description and instructions: 
### KernelOracle: Predicting the Linux Scheduler's Next Move with Deep Learning
Efficient task scheduling is paramount in the Linux kernel, where the Completely Fair Scheduler (CFS) meticulously manages CPU resources to balance high utilization with interactive responsiveness. This research pioneers the use of deep learning techniques to predict the sequence of tasks selected by CFS, aiming to evaluate the feasibility of a more generalized and potentially more adaptive task scheduler for diverse workloads. Our core contributions are twofold: first, the systematic generation and curation of a novel scheduling dataset from a running Linux kernel, capturing real-world CFS behavior; and second, the development, training, and evaluation of a Long Short-Term Memory (LSTM) network designed to accurately forecast the next task to be scheduled. Our paper further discusses the practical pathways and implications of integrating such a predictive model into the kernel's scheduling framework. The findings and methodologies presented herein open avenues for data-driven advancements in kernel scheduling, with the full source code provided for reproducibility and further exploration.

### Gathering data:

1. Use VagrantFile to spawn up an Ubuntu VM.
2. Inside the VM, install nginx and apache-utils (it contains ab).
3. Start nginx
4. Start the ab tool to initialize load tests.
5. Start perf for a few seconds and record the data.
6. A perf.data file will be generated. Execute the command ``` sudo perf sched map > out_ab_nginx.txt ```.
7. Come out of the VM. Shut it down.
8. Create ```data``` directory in the root of this repo.
9. Move the out_ab_nginx.txt to the data folder.
10. Run the data_preprocessing.py to convert ```out_ab_nginx.txt``` to ```scheduling_data_out_ab_nging.csv```.
11. Run ```conda env create -f environment.yml``` to install conda environment.
12. Activate the created environment.
13. Run ```python train.py``` to train the model.
