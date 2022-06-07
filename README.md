# parllel

An RL accelerator framework with reusable types for taking an existing RL training loop and achieving maximum utilization of existing computational resources.

## Getting Started

Create a new conda environment for this project and activate it.

```
conda create -n parllel
conda activate parllel
```

Install pytorch (or ML framework of your choice, coming soon). The process depends on your hardware, but some common cases are handled by installing yml files.

Linux with CUDA 11.3+: `conda env update --name parllel --file torch_cuda11.yml`

Mac OS on Apple Silicon: `conda env update --name parllel --file torch_m1.yml`

Install other requirements.

```
pip install -r requirements.txt
```

Install parllel repo itself.

```
pip install -e .
```

### Examples

To run the examples, you must also install the development requirements.

If installing `hera-gym` in development mode, do this now, before installing the development requirements. By default, a fresh copy is installed.

```
pip install -r requirements_dev.txt
```
