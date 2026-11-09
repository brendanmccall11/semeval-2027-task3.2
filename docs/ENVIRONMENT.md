# Environment

## GPU experiments (Google Colab)

- GPU: NVIDIA A100 80 GB (Gemma 4 31B in bfloat16 needs about 60 GB for inference and LoRA training).
- Python 3.13 (Colab default at the time).
- transformers 5.22.1 for Gemma scoring and LoRA training; notebook 11 ran with 5.22.0.
- peft 0.21.2. Colab's preinstalled `torchao` 0.10.0 conflicts with this peft version, so it was uninstalled before attaching the adapter (`pip uninstall -y torchao`). Nothing in this project uses torchao.
- `bcubed` must be installed for the official scorer (`pip install bcubed`).
- Exact package versions for each GPU run are saved with that run's outputs on Drive.

## Local harness tests

- `requirements.txt` is the environment used to verify the evaluation harness (Python 3.12, CPU only). It is not a record of the Colab GPU environment.
