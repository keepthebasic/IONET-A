# IO-Net-A

Code for probabilistic lake water-quality forecasting with incomplete observatory streams (IO-Net-A).

## Contents

- `models/` — IO-Net-A network definition
- `scripts/` — data preparation, training, evaluation, robustness probes, and figure helpers
- `configs/` — locked hyperparameters used in the reported experiments

## Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
export MKL_THREADING_LAYER=GNU
```

## Data

Raw LakeBeD tables are **not** redistributed in this repository. Download the public release, then build processed NPZ windows with the scripts below.

| Item | Detail |
|------|--------|
| Dataset | **LakeBeD-US: Computer Science Edition (LakeBeD-US-CSE)** |
| DOI | https://doi.org/10.57967/hf/3771 |
| Download | https://huggingface.co/datasets/eco-kgml/LakeBeD-US-CSE |
| Data tree | https://huggingface.co/datasets/eco-kgml/LakeBeD-US-CSE/tree/main/Data |
| Related paper | https://doi.org/10.5194/essd-17-3141-2025 |
| Ecology Edition (optional) | https://doi.org/10.6073/pasta/c56a204a65483790f6277de4896d7140 |

**This work uses** the CSE HighFrequency / LowFrequency parquet products, resampled to **4 h** windows (`seq_len=168`, 6-step / 24 h outlook):

- Missingness atlas: **13** lakes with a usable 4 h product  
- Mechanism deep-dive: **BVR**  
- Leave-one-lake-out: **ME, BVR, FCR, TR, SP**

Place HF/LF tables under a local layout such as `data/LakeBeD-US-CSE/Data/{HighFrequency,LowFrequency}/<LAKE>/`, then:

```bash
python scripts/data_process.py --help
```

Processed windows should live under a path such as `processed/BVR_4h/` (must contain `train.npz`, `val.npz`, `test.npz`, `meta.json`, `scalers.json`). Follow the LakeBeD-US license / citation terms when redistributing derived products.

## Reproduce core experiments (BVR, 4 h)

Phase A (imputation stress, LSTM / GRU-D):

```bash
python scripts/run_phase_a.py --proc-dir processed/BVR_4h --device cuda
```

Phase B (CRPS comparison: persistence, LSTM, Q-LSTM, GRU-D, IO-Net-A):

```bash
python scripts/run_phase_b.py --proc-dir processed/BVR_4h --device cuda
```

Lab thinning, probe leave-one-out, and leave-one-lake-out:

```bash
python scripts/run_lab_thinning.py --help
python scripts/run_probe_loo.py --help
python scripts/run_lolo.py --help
```

Single-model train / evaluate:

```bash
python scripts/train_ionet_a.py --proc-dir processed/BVR_4h --out-dir results/ionet_a_seed0 --seed 0
python scripts/evaluate_ionet_a.py --proc-dir processed/BVR_4h --results-dir results/ionet_a_seed0
```

Optional validation-set interval calibration:

```bash
python scripts/evaluate_ionet_a.py --proc-dir processed/BVR_4h --results-dir results/ionet_a_seed0 --calibrate
```

## Citation

If you use this code, please cite the accompanying Water Research manuscript on monitoring information value and IO-Net-A. Please also cite LakeBeD-US-CSE (DOI above) when using the data.
