# parllel

An RL accelerator framework with reusable types for taking an existing RL training loop and achieving maximum utilization of existing computational resources. This project is essentially a fork and complete refactor of rlpyt.

## Getting Started

Create a new conda environment for this project.

```
conda create -n parllel
conda activate parllel
```

Update conda environment with dependencies from yml file.

```
conda env update --name parllel --file torch.yml
```

Install parllel repo itself.

```
pip install -e .
```
