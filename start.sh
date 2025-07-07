
# clone this repository
git clone https://github.com/maxfleck/BayesForLaurence.git

# enter folder
cd BayesForLaurence

# install uv
# uv website:
# https://docs.astral.sh/uv/getting-started/installation/
# probably:
wget -qO- https://astral.sh/uv/install.sh | sh

# install python and python packages
uv python install 3.8
uv sync

# get EDBO+
git clone https://github.com/doyle-lab-ucla/edboplus.git

# run tool
uv run marimo edit # or directly start.py
