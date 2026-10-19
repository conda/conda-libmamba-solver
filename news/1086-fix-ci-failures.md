### Other

* Keep Python 3.10 test coverage against conda 26.9.0 and run conda main tests with Python 3.11 or newer. (#1086)
* Fix `test_pytorch_gpu` by installing the `nvidia-virtual-packages` conda plugin and setting `CONDA_OVERRIDE_CUDA_ARCH`, since recent conda-forge CUDA builds require the `__cuda_arch` virtual package. (#1086)
