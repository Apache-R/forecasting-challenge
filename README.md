# Cloud Workload Forecasting Challenge

This repository contains a reproducible forecasting solution for the fixed
`data-v1` release of the Microsoft Azure Public Dataset V2 derivative.

## Setup

Python 3.11 or newer is required.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The editable project installation is also supported:

```powershell
python -m pip install -e ".[dev]"
```

## Dataset

Download `data-v1.tar.gz` from the [data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)
and extract it into `challenge_data\`. Verify it before use:

```powershell
cd challenge_data
sha256sum -c SHA256SUMS
cd ..
```

The release contains 10,000 VMs, 134 observations per VM, and five-minute
sampling. See [DATASET.md](DATASET.md) for the field contract and
[DATA_CARD_TEMPLATE.md](DATA_CARD_TEMPLATE.md) for provenance.

## Generate the required submission

```powershell
python src\predict.py --data challenge_data --output predictions.csv
```

The command constructs the two required aggregate targets, fits only on the
first 110 timestamps, forecasts the final 24 timestamps, validates the exact
submission shape, and writes `predictions.csv` without an index.

## Validate the approach

Run leakage-safe expanding-window evaluation on the pre-holdout data:

```powershell
python src\evaluate.py --data challenge_data --output reports\validation.csv
```

Because this challenge publishes the holdout labels, you can separately
inspect the final prediction score after generating `predictions.csv`:

```powershell
python src\evaluate.py --data challenge_data --public-holdout --predictions predictions.csv
```

This is an analysis-only score, not a model-selection score. It compares the
submitted forecast with the public labels and reports the persistence baseline.
Generate report figures and the same score with:

```powershell
python src\make_report_assets.py --data challenge_data --predictions predictions.csv
```

Run tests:

```powershell
python -m pytest -q
```

## Implementation

- [`src/data.py`](src/data.py): ingestion, validation, aggregation, and
  chronological splitting.
- [`src/features.py`](src/features.py): lag and rolling CPU features.
- [`src/model.py`](src/model.py): persistence fleet forecast and recursive
  Ridge CPU forecast.
- [`src/evaluate.py`](src/evaluate.py): expanding-window baseline comparison.
- [`src/make_report_assets.py`](src/make_report_assets.py): report PNGs and
  public-holdout scoring.
- [`src/predict.py`](src/predict.py): reproducible submission CLI.
- [`reports/final_report.md`](reports/final_report.md): analysis,
  methodology, validation, limitations, and operational recommendation.

The notebooks are exploratory test artifacts and are not required to generate
the submission.

## Source and attribution

The data is a curated derivative of the
[Microsoft Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset).
See [SOURCES.md](SOURCES.md), [DATA_RELEASE.md](DATA_RELEASE.md), and the
upstream license and usage terms.
