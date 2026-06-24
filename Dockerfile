# Reproducible environment for the Mt. Rainier seismic-hydrology analysis.
# Uses the pinned pixi.lock so the container matches the development environment
# byte-for-byte. Build:  docker build -t seis-hydro .
# Run (offline figures):  docker run --rm seis-hydro pixi run make figures-from-cache
FROM ghcr.io/prefix-dev/pixi:0.41.4 AS base

WORKDIR /app

# Install the locked environment first (cached unless the lock changes)
COPY pixi.toml pixi.lock ./
RUN pixi install --locked

# Bring in the project
COPY . .

# Default: drop into the project env; override the command to run a target, e.g.
#   docker run --rm seis-hydro pixi run make figures-from-cache
CMD ["pixi", "run", "bash"]
