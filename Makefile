# Tout se règle par variables : make start WORKERS=20 MAX_SECONDS=300 RUN=runs/autre
WORKERS ?= $(shell nproc)
RANDOM_TRIALS ?= $(shell echo $$(( $(WORKERS) * 50 )))
RUN ?= runs/horon
STATES ?= states/horon-endgame.state
# durée max d'une partie NEAT (secondes de jeu)
MAX_SECONDS ?= 120
# arrêt après N secondes sans rien de nouveau
PATIENCE ?= 15
# durée max d'une partie aléatoire
RANDOM_SECONDS ?= 20
# vide = sans limite
GENERATIONS ?=
# options en plus pour train.py
ARGS ?=
# partie solo (joueur seul depuis Horon) : durée, fréquence dans make start (générations)
SOLO_SECONDS ?= 1200
SOLO_EVERY ?= 5
GENOME ?= models/neat.pkl
# RANDOM_LOCATION=1 : replay, compare, ppo-replay et ppo-solo partent d'une cellule de DATASET_ARCHIVE
# tirée au hasard (la même pour les deux joueurs de make compare)
RANDOM_LOCATION ?=
# make compare : durée des deux parties en fenêtre (secondes de jeu, vitesse réelle)
COMPARE_SECONDS ?= 120

# PPO : un seul réseau. Python avec torch : le venv ROCm s'il existe, sinon .venv (make install-ppo)
PPO_PY ?= $(firstword $(wildcard /mnt/data/dev/trainvenv-rocm/bin/python) .venv/bin/python)
PPO_RUN ?= runs/ppo
ARCHIVE ?= $(RUN)/archive
PPO_WORKERS ?= $(shell echo $$(( $(WORKERS) > 1 ? $(WORKERS) - 1 : 1 )))
ENVS ?= 4
DEMOS ?= 4
HORON ?= 0.2
SOLO_MINUTES ?= 15
PPO_MAX_SECONDS ?= 1200
PPO_PATIENCE ?= 120
PPO_ARGS ?=
# imitation des démos : poids dans la perte PPO, pas à chaque nouvelle démo (0 = PPO pur sur le curriculum)
BC_COEF ?= 0.0
BC_BURST ?= 0
MODEL ?= models/ppo.pt
CHECK ?= 20
# dataset world model
DATASET ?= /mnt/data/datasets/zelda
DATASET_GB ?= 50
# cœurs libres au lancement : nproc moins la charge moyenne sur 1 min, au moins 1
DATASET_WORKERS ?= $(shell awk -v n=$$(nproc) '{w = n - int($$1 + 0.5); print (w < 1 ? 1 : w)}' /proc/loadavg)
# modèles PPO, à parts égales dans les parties jouées par un modèle ; vide = tout au hasard
DATASET_MODEL ?= models/ppo.pt models/ppo-kills.pt
# part des parties jouées par le modèle, le reste au hasard
DATASET_MODEL_FRAC ?= 0.7
# cellules de départ (8 par salle au plus, tirées de runs/horon/archive)
DATASET_ARCHIVE ?= spawns

# pour make pull : user@serveur
SERVER ?=
REMOTE_DIR ?= zelda-neat
TELEVAULT ?= $(HOME)/dev/televault/televault
DIST = dist/neat-$(shell date +%F).tar.gz

PY = .venv/bin/python
UNIT = zelda-explore
PPO_UNIT = zelda-ppo
DATASET_UNIT = zelda-dataset
# Départ des parties regardées : Horon, ou une cellule au hasard tirée une seule fois par make.
ifneq ($(strip $(RANDOM_LOCATION)),)
START := $(shell $(PPO_PY) -c "import random, sys; sys.path.insert(0, '.'); from dataset import room_capped; print(random.choice(room_capped('$(DATASET_ARCHIVE)', 8)))")
else
START = $(firstword $(wildcard $(STATES)))
endif
MAKE_DATASET = $(PPO_PY) dataset.py --archive $(DATASET_ARCHIVE) --out $(DATASET) --max-gb $(DATASET_GB) \
	--workers $(DATASET_WORKERS) --start $(firstword $(wildcard $(STATES))) \
	--model $(DATASET_MODEL) --model-frac $(DATASET_MODEL_FRAC)
PPO_RESUME = $(if $(wildcard $(PPO_RUN)/ckpt.pt),--resume,)
PPO = $(PPO_PY) ppo.py --archive $(ARCHIVE) --run $(PPO_RUN) --start $(firstword $(wildcard $(STATES))) \
	--workers $(PPO_WORKERS) --envs-per-worker $(ENVS) --demos $(DEMOS) --horon $(HORON) \
	--max-seconds $(PPO_MAX_SECONDS) --patience $(PPO_PATIENCE) \
	--solo-minutes $(SOLO_MINUTES) --solo-seconds $(SOLO_SECONDS) \
	--bc-coef $(BC_COEF) --bc-burst $(BC_BURST) $(PPO_ARGS)
# Reprise automatique si le run a déjà un checkpoint, sinon nouveau run.
RESUME = $(if $(wildcard $(RUN)/ckpt-*),--resume,)
# NEAT en score solo depuis des cellules au hasard (make neat-spawns-start)
NEAT_RUN ?= runs/neat-spawns
NEAT_STARTS ?= 5
NEAT_UNIT = zelda-neat-spawns
NEAT_SPAWNS = $(PY) train.py --run $(NEAT_RUN) --states $(STATES) --spawns $(DATASET_ARCHIVE) \
	--workers $(WORKERS) --starts $(NEAT_STARTS) --max-seconds $(MAX_SECONDS) --patience $(PATIENCE) \
	--solo-every $(SOLO_EVERY) --solo-seconds $(SOLO_SECONDS) $(if $(wildcard $(NEAT_RUN)/ckpt-*),--resume,) $(ARGS)
# Exploration Go-Explore par parties aléatoires (make start). NEAT reste disponible : make neat.
EXPLORE = $(PY) explore.py --run $(RUN) --states $(STATES) --workers $(WORKERS) \
	--trials $(RANDOM_TRIALS) --seconds $(RANDOM_SECONDS) --patience $(PATIENCE) $(ARGS)
TRAIN = $(PY) train.py --run $(RUN) --states $(STATES) --workers $(WORKERS) \
	--random-trials $(RANDOM_TRIALS) --max-seconds $(MAX_SECONDS) --patience $(PATIENCE) \
	--random-seconds $(RANDOM_SECONDS) --solo-every $(SOLO_EVERY) --solo-seconds $(SOLO_SECONDS) $(if $(strip $(GENERATIONS)),--generations $(GENERATIONS),) $(ARGS)

.PHONY: help install bench run neat start start-tmux stop status log replay compare reset package vault pull \
	solo paths paths-check install-ppo ppo ppo-start ppo-stop ppo-log ppo-solo ppo-replay ppo-reset dataset dataset-start dataset-stop neat-spawns neat-spawns-start neat-spawns-stop neat-spawns-log

help:
	@echo "make install      crée .venv"
	@echo "make bench        mesure le bon nombre de workers"
	@echo "make run          explore l'archive par parties aléatoires, au premier plan (reprend l'archive)"
	@echo "make start        pareil en tâche de fond (systemd), log dans RUN/train.log"
	@echo "make neat         ancien entraînement NEAT + Go-Explore, au premier plan"
	@echo "make start-tmux   pareil dans une session tmux 'neat' (sans systemd)"
	@echo "make stop         arrêt propre avec checkpoint"
	@echo "make status       génération, salles, kills"
	@echo "make log          suit la progression"
	@echo "make replay       regarde GENOME (défaut models/neat.pkl) jouer depuis Horon (fenêtre)"
	@echo "make compare      NEAT (GENOME) et PPO (MODEL) jouent en même temps depuis Horon, deux fenêtres"
	@echo "                  RANDOM_LOCATION=1 : départ dans une cellule au hasard (replay, compare, ppo-replay, ppo-solo)"
	@echo "make reset        efface RUN (demande CONFIRM=oui)"
	@echo "make package      archive du projet + RUN dans dist/"
	@echo "make vault        package + envoi dans le Televault"
	@echo "make pull         rapatrie RUN depuis SERVER=user@serveur"
	@echo "make solo         GENOME (défaut models/neat.pkl) joue seul depuis Horon, SOLO_SECONDS"
	@echo "make paths        cellules de l'archive joignables depuis Horon par un chemin d'inputs"
	@echo "make paths-check  rejoue CHECK chemins depuis Horon et vérifie l'arrivée"
	@echo "make install-ppo  installe torch (CPU) dans .venv, si pas de venv ROCm"
	@echo "make ppo          entraîne le joueur PPO (écran + LSTM + imitation) au premier plan"
	@echo "make ppo-start    pareil en tâche de fond (systemd), log dans PPO_RUN/train.log"
	@echo "make ppo-stop     arrêt propre du PPO avec checkpoint"
	@echo "make ppo-log      suit le PPO"
	@echo "make ppo-solo     MODEL (défaut models/ppo.pt) joue seul depuis Horon"
	@echo "make ppo-replay   regarde MODEL jouer depuis Horon (fenêtre)"
	@echo "make ppo-reset    efface PPO_RUN (demande CONFIRM=oui)"
	@echo "make neat-spawns-start / -stop / -log   NEAT en score solo, départs au hasard dans spawns/, run NEAT_RUN"
	@echo "                  meilleur génome solo dans NEAT_RUN/best-solo.pkl (make compare GENOME=...)"
	@echo "make dataset      enregistre des parties (écran + boutons) dans DATASET jusqu'à DATASET_GB Go"
	@echo "make dataset-start / dataset-stop   pareil en tâche de fond"
	@echo "variables : WORKERS RANDOM_TRIALS RUN STATES MAX_SECONDS PATIENCE RANDOM_SECONDS GENERATIONS ARGS"
	@echo "            SOLO_SECONDS SOLO_EVERY GENOME COMPARE_SECONDS RANDOM_LOCATION"
	@echo "            PPO_RUN ARCHIVE PPO_WORKERS ENVS DEMOS HORON SOLO_MINUTES BC_COEF BC_BURST PPO_ARGS MODEL PPO_PY\n            DATASET DATASET_GB DATASET_WORKERS DATASET_MODEL DATASET_MODEL_FRAC DATASET_ARCHIVE"

install:
	./install.sh

bench:
	$(PY) bench_workers.py

run:
	$(EXPLORE)

neat:
	$(TRAIN) $(RESUME)

start:
	-systemctl --user reset-failed $(UNIT) 2>/dev/null
	mkdir -p $(RUN)
	systemd-run --user --unit $(UNIT) --working-directory=$(CURDIR) --setenv=PYTHONUNBUFFERED=1 \
	  sh -c 'exec $(EXPLORE) >> $(RUN)/train.log 2>&1'
	@echo "exploration lancée : $(WORKERS) workers, $(RANDOM_TRIALS) parties aléatoires de $(RANDOM_SECONDS) s par tour"

start-tmux:
	mkdir -p $(RUN)
	tmux new-session -d -s neat "PYTHONUNBUFFERED=1 $(EXPLORE) 2>&1 | tee -a $(RUN)/train.log"
	@echo "lancé dans tmux : 'tmux attach -t neat' pour voir, Ctrl+B puis D pour repartir"

stop:
	@if systemctl --user is-active -q $(UNIT) 2>/dev/null; then \
	  systemctl --user kill -s INT $(UNIT); \
	  while systemctl --user is-active -q $(UNIT); do sleep 1; done; \
	elif tmux has-session -t neat 2>/dev/null; then \
	  tmux send-keys -t neat C-c; \
	  while tmux has-session -t neat 2>/dev/null; do sleep 1; done; \
	else echo "rien ne tourne"; fi
	@echo arrêté

status:
	@printf "service : "; systemctl --user is-active $(UNIT) 2>/dev/null || (tmux has-session -t neat 2>/dev/null && echo tmux) || true
	@grep -E "^tour|Running generation" $(RUN)/train.log | tail -1 | tr -d '*' | sed 's/^ *Running/génération en cours :/'
	@grep -E "^archive :|^kills" $(RUN)/train.log | tail -2
	@grep -c "^plateau" $(RUN)/train.log | sed 's/^/redémarrages sur plateau : /'
	@grep "^solo" $(RUN)/train.log | tail -1
	@grep -q Traceback $(RUN)/train.log && echo "ATTENTION : un Traceback est dans le log" || true
	@if [ -f $(PPO_RUN)/train.log ]; then \
	  printf "ppo : "; systemctl --user is-active $(PPO_UNIT) 2>/dev/null || true; \
	  grep -E "^ppo :|^curriculum" $(PPO_RUN)/train.log | tail -2; \
	  grep "^solo" $(PPO_RUN)/train.log | tail -1 | sed 's/^solo/solo ppo/'; \
	  [ -f $(PPO_RUN)/solo-best.json ] && sed 's/^/meilleur solo ppo : /' $(PPO_RUN)/solo-best.json && echo; \
	  grep -q Traceback $(PPO_RUN)/train.log && echo "ATTENTION : un Traceback est dans le log PPO" || true; \
	fi
	@[ -f $(DATASET)/dataset.log ] && grep "^dataset" $(DATASET)/dataset.log | tail -1 || true

log:
	tail -f $(RUN)/train.log | grep --line-buffered -E "^tour|Running generation|archive :|kills|Generation time|plateau|solo|Traceback"

replay:
	$(PPO_PY) solo.py --genome $(GENOME) --state $(START) --seconds $(COMPARE_SECONDS) --window

compare:
	$(PPO_PY) solo.py --genome $(GENOME) --state $(START) --seconds $(COMPARE_SECONDS) --window & \
	$(PPO_PY) solo.py --model $(MODEL) --state $(START) --seconds $(COMPARE_SECONDS) --window & \
	wait

reset:
	@[ "$(CONFIRM)" = oui ] || { echo "efface $(RUN) : relance avec make reset CONFIRM=oui"; exit 1; }
	-$(MAKE) --no-print-directory stop
	rm -rf $(RUN)
	@echo "$(RUN) effacé, make start relance un nouveau run"

package:
	mkdir -p dist
	tar cf dist/tmp.tar -C .. --exclude='$(notdir $(CURDIR))/.venv' --exclude='__pycache__' \
	  --exclude='$(notdir $(CURDIR))/dist' --exclude='$(notdir $(CURDIR))/runs' --exclude='*.tmp' \
	  $(notdir $(CURDIR))
	tar rf dist/tmp.tar -C .. --exclude='*.tmp' $(notdir $(CURDIR))/$(RUN)
	gzip -c dist/tmp.tar > $(DIST) && rm dist/tmp.tar
	@ls -la $(DIST)

vault: package
	zsh -ic '$(TELEVAULT) up $(CURDIR)/$(DIST)'

pull:
	@[ -n "$(strip $(SERVER))" ] || { echo "make pull SERVER=user@serveur [REMOTE_DIR=zelda-neat]"; exit 1; }
	@[ ! -e $(RUN) ] || mv $(RUN) $(RUN)-avant-pull-$(shell date +%s)
	rsync -a $(SERVER):$(REMOTE_DIR)/$(RUN)/ $(RUN)/
	@echo "$(RUN) rapatrié, make start pour reprendre"

solo:
	$(PY) solo.py --genome $(GENOME) --state $(firstword $(wildcard $(STATES))) --seconds $(SOLO_SECONDS)

paths:
	$(PY) paths.py --archive $(ARCHIVE)

paths-check:
	$(PY) paths.py --archive $(ARCHIVE) --check $(CHECK) --workers $(WORKERS)

install-ppo:
	uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu

ppo:
	$(PPO) $(PPO_RESUME)

ppo-start:
	-systemctl --user reset-failed $(PPO_UNIT) 2>/dev/null
	mkdir -p $(PPO_RUN)
	systemd-run --user --unit $(PPO_UNIT) --working-directory=$(CURDIR) --setenv=PYTHONUNBUFFERED=1 \
	  sh -c 'exec $(PPO) $(PPO_RESUME) >> $(PPO_RUN)/train.log 2>&1'
	@echo "PPO lancé : $(PPO_WORKERS) workers x $(ENVS) émulateurs, archive $(ARCHIVE), $(if $(PPO_RESUME),reprise,nouveau run)"

ppo-stop:
	@if systemctl --user is-active -q $(PPO_UNIT) 2>/dev/null; then \
	  systemctl --user kill -s INT $(PPO_UNIT); \
	  while systemctl --user is-active -q $(PPO_UNIT); do sleep 1; done; echo arrêté; \
	else echo "le PPO ne tourne pas"; fi

ppo-log:
	tail -f $(PPO_RUN)/train.log | grep --line-buffered -v -E "amdgpu.ids|SDL2"

ppo-solo:
	$(PPO_PY) solo.py --model $(MODEL) --state $(START) --seconds $(SOLO_SECONDS)
	$(PPO_PY) solo.py --model $(MODEL) --state $(START) --seconds $(SOLO_SECONDS) --greedy

ppo-replay:
	$(PPO_PY) solo.py --model $(MODEL) --state $(START) --seconds $(SOLO_SECONDS) --window

ppo-reset:
	@[ "$(CONFIRM)" = oui ] || { echo "efface $(PPO_RUN) : relance avec make ppo-reset CONFIRM=oui"; exit 1; }
	-$(MAKE) --no-print-directory ppo-stop
	rm -rf $(PPO_RUN)
	@echo "$(PPO_RUN) effacé"

neat-spawns:
	$(NEAT_SPAWNS)

neat-spawns-start:
	-systemctl --user reset-failed $(NEAT_UNIT) 2>/dev/null
	mkdir -p $(NEAT_RUN)
	systemd-run --user --unit $(NEAT_UNIT) --working-directory=$(CURDIR) --setenv=PYTHONUNBUFFERED=1 \
	  sh -c 'exec $(NEAT_SPAWNS) >> $(NEAT_RUN)/train.log 2>&1'
	@echo "NEAT lancé dans $(NEAT_RUN) : départs au hasard, score solo, $(WORKERS) workers"

neat-spawns-stop:
	@if systemctl --user is-active -q $(NEAT_UNIT) 2>/dev/null; then \
	  systemctl --user kill -s INT $(NEAT_UNIT); \
	  while systemctl --user is-active -q $(NEAT_UNIT); do sleep 1; done; echo arrêté; \
	else echo "NEAT spawns ne tourne pas"; fi

neat-spawns-log:
	tail -f $(NEAT_RUN)/train.log | grep --line-buffered -E "^départs|^solo|Running generation|Traceback"

dataset:
	$(MAKE_DATASET)

dataset-start:
	-systemctl --user reset-failed $(DATASET_UNIT) 2>/dev/null
	mkdir -p $(DATASET)
	systemd-run --user --unit $(DATASET_UNIT) --working-directory=$(CURDIR) --setenv=PYTHONUNBUFFERED=1 \
	  sh -c 'exec $(MAKE_DATASET) >> $(DATASET)/dataset.log 2>&1'
	@echo "dataset lancé dans $(DATASET), modèle : $(or $(strip $(DATASET_MODEL)),aucun (aléatoire))"

dataset-stop:
	-systemctl --user kill -s INT $(DATASET_UNIT)
