# install uv
# get repo
uv python install 3.8
uv sync
git clone https://github.com/doyle-lab-ucla/edboplus.git
uv run marimo edit # or directly start.py
