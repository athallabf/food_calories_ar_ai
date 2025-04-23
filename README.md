# Make sure UV is installed

https://docs.astral.sh/uv/getting-started/installation/

this project uses python 3.10.17
to install use
`uv python install 3.10 --default --preview`
make sure PATH is configured
`uv tool update-shell`

1. uv venv --python 3.10
2. source .venv/bin/activate
3. uv add opencv-python numpy
4. uv run food_calories_ar.py
