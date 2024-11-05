ANTONIO - Abstract iNterpreTation fOr Nlp verIficatiOn
========

For detailed information of the design, refer to [ANTONIO: Towards a Systematic Method of Generating NLP Benchmarks for Verification](https://easychair.org/publications/paper/9ZGS) and [NLP Verification: Towards a General Methodology for Certifying Robustness](https://arxiv.org/abs/2403.10144).
ANTONIO has also been used to produce the [safeNLP](https://github.com/ANTONIONLP/safeNLP) benchmark used in [VNN-COMP 2024](https://sites.google.com/view/vnn2024).

Structure
------------
```
.
├── datasets
│   ├── medical
│   │   └── data                                     - folder containing the Medical dataset
│   │
│   └── ruarobot
│       └── data                                     - folder with the R-U-A-Robot dataset
│   
├── src
│   ├── data.py                                      - file for loading and processing the data
│   ├── example.py                                   - file containing an example running the full pipeline
│   ├── hyperrectangles.py                           - file for creating hyper-rectangles
│   ├── perturbations.py                             - file for creating, saving and embedding the perturbations
│   ├── property_parser.py                           - file for creating the properties in VNNlib or Marabou formats
│   ├── results.py                                   - file for calculating results
│   └── train.py                                     - file for training the networks
│
├── requirements.txt                                 - pip requirements
└── tf2onnx.sh                                       - script for creating onnx networks from tensorflow
```

System Requirements
------------
Ubuntu 18.04 (64-bit), Python 3.6 or higher.

Installation
------------
Install the dependencies:
```
pip3 install -r requirements.txt
```
The character and word-level perturbations implemented do not require any further installation. To use the sentence-level perturbations implemented install:
* [Replicate](https://replicate.com/) (This API requires a subscription)

To verify the hyper-rectangles also install a verifier:
* [Marabou](https://github.com/NeuralNetworkVerification/Marabou)
* [ERAN](https://github.com/eth-sri/eran)
* [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN/tree/main)

Instructions
-------------
To run the example:
```
python3 src/example.py
```
