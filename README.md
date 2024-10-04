[//]: # (>📋  A template README.md for code accompanying a Machine Learning paper)

# DapperFL: Domain Adaptive Federated Learning with Model Fusion Pruning for Edge Devices

<p align="center">
  <a href="https://github.com/FedML-AI/FedML/projects/1"><img alt="Roadmap" src="https://img.shields.io/badge/roadmap-FedML-informational.svg?style=flat-square"></a>
  <a href="#"><img alt="Python3" src="https://img.shields.io/badge/Python-3-brightgreen.svg?style=flat-square"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E1.9-orange"></a>
</p>

This repository is the official implementation of [**DapperFL: Domain Adaptive Federated Learning
with Model Fusion Pruning for Edge Devices**]. (Accepted by **NeurIPS 2024**)


[//]: # (>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials)
<img src="./docs/DapperFL_overview.jpg" width = "80%" height = "" alt="DapperFL" TITLE="Overview of DapperFL." />


## Requirements

To install requirements:
```setup
pip install -r requirements.txt
```

We use <a href="https://wandb.ai/">wandb</a> to keep a log of our experiments.
If you don't have a wandb account, just install it and use it as offline mode.
```wandb
pip install wandb
wandb off
```

[//]: # (>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...)

## Training & Evaluation

To train the model(s) in the paper, run this command:
```train
python ./fedml_experiments/standalone/domain_generalization/main.py \
       --model dapperfl 
       --dataset fl_officecaltech 
       --backbone resnet18
```

[//]: # (>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.)

## Arguments

You can modify the arguments to run DapperFL on other settings. The arguments are described as follows:

| Arguments             | Description                                                                                                       |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|
| `prefix`              | A prefix for logging.                                                                                             |
| `communication_epoch` | Total communication rounds of Federated Learning.                                                                 |
| `local_epoch`         | Local epochs for local model updating.                                                                            |
| `parti_num`           | Number of participants.                                                                                           |
| `model`               | Name of FL framework.                                                                                             |
| `dataset`             | Datasets used in the experiment. Options: `fl_officecaltech`, `fl_digits`.                                        |
| `pr_strategy`         | Pruning ratio used to prune local models. Options: `0` (without pruning), `0.1` ~ `0.9`, `AD` (adaptive pruning). |
| `backbone`            | Backbone global model. Options: `resnet10`, `resnet18`.                                                           |
| `alpha`               | Coefficient alpha in co-pruning. Default: `0.9`.                                                                  |
| `alpha_min`           | Coefficient alpha_min in co-pruning. Default: `0.1`.                                                              |
| `epsilon`             | Coefficient epsilon in co-pruning. Default: `0.2`.                                                                |
| `reg_coeff`           | Coefficient for L2 regularization. Default: `0.01`.                                                               |                                                                     |
| `seed`                | Random seed.                                                                                                      |

[//]: # (>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results &#40;section below&#41;.)

[//]: # (## Pre-trained Models)

[//]: # ()
[//]: # (You can download pretrained models here:)

[//]: # ()
[//]: # (- [My awesome model]&#40;https://drive.google.com/mymodel.pth&#41; trained on ImageNet using parameters x,y,z. )

[//]: # ()
[//]: # (>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained &#40;if applicable&#41;.  Alternatively you can have an additional column in your results table with a link to the models.)

[//]: # (## Results)

[//]: # ()
[//]: # (Our model achieves the following performance on :)

[//]: # ()
[//]: # (### [Image Classification on ImageNet]&#40;https://paperswithcode.com/sota/image-classification-on-imagenet&#41;)

[//]: # ()
[//]: # (| Model name         | Top 1 Accuracy  | Top 5 Accuracy |)

[//]: # (| ------------------ |---------------- | -------------- |)

[//]: # (| My awesome model   |     85%         |      95%       |)

[//]: # ()
[//]: # (>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. )


[//]: # (## Contributing)

[//]: # ()
[//]: # (>📋  Pick a licence and describe how to contribute to your code repository. )
