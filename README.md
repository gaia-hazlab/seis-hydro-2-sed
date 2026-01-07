# seis-hydro-2-sed

Turn river discharge and seismic power to predict dynamic bedload transport.

## Description

This project combines seismological and hydrological data to predict dynamic bedload transport in rivers. It uses seismic power measurements and river discharge data to model sediment movement.

## Installation Instructions

### Prerequisites

This project requires Python 3.9 or higher.

### Option 1: Using pixi (Recommended)

[Pixi](https://pixi.sh) is a modern, fast package manager that handles all dependencies automatically.

1. Install pixi following the [official installation guide](https://pixi.sh/latest/#installation)

2. Clone and set up the environment:
```bash
git clone https://github.com/gaia-hazlab/seis-hydro-2-sed.git
cd seis-hydro-2-sed
pixi install
```

3. Activate the environment:
```bash
pixi shell
```

4. For Jupyter notebooks, the Python kernel will be available at `.pixi/envs/default/bin/python`

### Option 2: Using conda/mamba

If you prefer using conda or mamba:

1. Create the environment from the environment file:
```bash
conda env create -f environment.yml
```

Or with mamba (faster):
```bash
mamba env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate seis-hydro-2-sed
```

### Option 3: Using pip

For pip-based installation (requires system dependencies):

1. **macOS users:** Install GDAL first
```bash
brew install gdal
```

2. **Linux users:** Install system dependencies
```bash
# Ubuntu/Debian
sudo apt-get install libgdal-dev

# Fedora
sudo dnf install gdal-devel
```

3. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Dependencies

The project uses the following main dependencies:

- **obspy** (>=1.4.2): Seismological data processing
- **matplotlib** (>=3.8.0): Plotting and visualization
- **numpy** (>=1.24.0): Numerical computing
- **scipy** (>=1.11.0): Scientific computing and signal processing

## Usage

After installation, you can import the required packages:

```python
import obspy
import matplotlib.pyplot as plt
import numpy as np
import scipy
```

## Batch workflow (generate plots + CSVs for all stations)

The notebook workflow in `notebooks/uw_river_rumble_dec2025.ipynb` can be run in batch mode over **all seismic stations that have an associated USGS gage** in the GAIA metadata table.

Run from the repo root:

```bash
pixi run python scripts/run_river_rumble_batch.py \
	--start 2025-12-01T00:00:00 \
	--end   2025-12-24T00:00:00 \
	--network-filter CC \
	--use-rss \
	--exclude-earthquakes \
	--clip-impulses \
	--despike-proxy
```

Outputs:
- Time series CSVs and `fit_parameters.csv` are written under `notebooks/data/results/`.
- Figures are written under `notebooks/figures/`.


## Development

To install development dependencies:

```bash
# With pip
pip install -e ".[dev]"

# With pixi, the ipykernel is already included
```


## Literature review
* Cook et al, 2020 https://onlinelibrary.wiley.com/doi/full/10.1002/esp.4993
* Cook et al, 2028: https://www.science.org/doi/full/10.1126/science.aat4981

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
