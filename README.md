## Acknowledgment 
This project builds upon Sampanna Yashwant Kahu's project, found at: [repository](https://github.com/SampannaKahu/KernelOracle), [paper](https://arxiv.org/abs/2505.15213).

## Abstract
Process scheduling is one of the most performance-critical
components of an operating system. The Linux Completely
Fair Scheduler (CFS) makes decisions purely from instantaneous
kernel state, with no memory of historical execution
patterns. KernelOracle [ 5 ] demonstrated that scheduling sequences
are learnable from traces using an LSTM network, but
the model was trained on only a single workload and inference
latency was far too high for real-time use. The goal of this
project is to address both, building off of this repository [4 ].
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

Full paper can be found [here](https://github.com/mmarcotullio/KernelOracle/blob/main/report.pdf). 


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
