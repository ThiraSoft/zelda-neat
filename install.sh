#!/bin/sh
# Installe l'environnement dans .venv. Demande uv (https://docs.astral.sh/uv/) ou un python3.12.
set -e
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
    uv venv -p 3.12 .venv
    uv pip install -p .venv/bin/python -r requirements.txt
else
    python3.12 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python -c "import pyboy, neat; print('ok : pyboy', pyboy.__version__ if hasattr(pyboy, '__version__') else '', 'neat-python')"
[ -f rom/oos.gbc ] || echo "attention : copie la ROM dans rom/oos.gbc"
