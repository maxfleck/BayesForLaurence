# BayesForLaurence

## Quickstart

i) get this repository
```terminal
# clone this repository
git clone https://github.com/maxfleck/BayesForLaurence.git

# enter folder
cd BayesForLaurence
```

ii) get uv
```terminal
# install uv
# uv website:
# https://docs.astral.sh/uv/getting-started/installation/
# probably:
wget -qO- https://astral.sh/uv/install.sh | sh
```

iii) install python, python packages and get EDBO+
```terminal
# install python and python packages
uv python install 3.8
uv sync

# get EDBO+
git clone https://github.com/doyle-lab-ucla/edboplus.git
```

iv) run the tool
```terminal

# run tool as user
uv run marimo run EDBOplus.py

# run tool as advanced user
uv run marimo edit EDBOplus.py

# run tool as developer
uv run marimo edit


```