# parllel

An RL accelerator framework with reusable types for taking an existing RL training loop and achieving maximum utilization of existing computational resources.

## Getting Started

Create a new conda environment for this project and activate it.

```
conda create -n parllel python=3.9
conda activate parllel
```

Install pytorch (or ML framework of your choice, coming soon). The process depends on your hardware, but some common cases are handled by installing yml files.

Linux with CUDA 11.3+: `conda env update --file torch_cuda11.yml`

Mac OS on Apple Silicon: `conda env update --file torch_m1.yml`

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

```
pip install -r requirements_dev.txt
```

If you already had `hera-gym` installed in development mode, you will now need to reinstall it, as it has been replaced by a fresh copy of hera_gym from gitlab.