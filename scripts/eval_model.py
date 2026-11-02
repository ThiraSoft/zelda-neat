#!/usr/bin/env python3
"""Évalue un modèle PPO (.pt) ou un génome NEAT (.pkl) sur des parties téléportées à graines fixes,
les mêmes pour les deux, comparables d'un run à l'autre (voir zneat/evaluate.py).

    eval_model.py models/ppo.pt [--games 30] [--seconds 120] [--out eval.json]
    eval_model.py runs/neat/best-eval.pkl
"""
import argparse
import json
import sys

sys.path.insert(0, ".")

from zneat.archive import read_state
from zneat.env import ZeldaEnv
from zneat.evaluate import EVAL_GAMES, EVAL_SECONDS, eval_game, summarize

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--games", type=int, default=EVAL_GAMES)
ap.add_argument("--seconds", type=float, default=EVAL_SECONDS)
ap.add_argument("--config", default="config-neat.ini")
ap.add_argument("--out")
args = ap.parse_args()

env = ZeldaEnv("rom/oos.gbc")
state = read_state("states/horon-endgame.state")
if args.model.endswith(".pkl"):
    import pickle

    import neat

    from zneat.runner import make_net, net_policy
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)
    genome = pickle.load(open(args.model, "rb"))
    policies = (net_policy(make_net(genome, config)) for _ in range(args.games))
    render = False
else:
    import torch
    torch.set_num_threads(1)
    from zneat.ppo_model import load_model, torch_policy
    model = load_model(args.model)
    policies = (torch_policy(model, seed=g) for g in range(args.games))
    render = True   # le PPO regarde l'écran
res = summarize([eval_game(env, p, state, g, args.seconds, render) for g, p in enumerate(policies)])
print(json.dumps(res, ensure_ascii=False))
if args.out:
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
