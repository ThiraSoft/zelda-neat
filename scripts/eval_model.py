#!/usr/bin/env python3
"""Évalue un modèle PPO sur des parties téléportées à graines fixes, comparables d'un run à l'autre.

    eval_model.py runs/iter1/model.pt [--games 30] [--seconds 120] [--out runs/iter1/eval.json]
"""
import argparse
import json
import random
import sys
from collections import Counter

sys.path.insert(0, ".")
import torch

from zneat import ram
from zneat.archive import read_state
from zneat.env import EQUIP_A, EQUIP_B, ZeldaEnv
from zneat.ppo_model import load_model, torch_policy
from zneat.runner import solo

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--games", type=int, default=30)
ap.add_argument("--seconds", type=float, default=120)
ap.add_argument("--out")
args = ap.parse_args()
torch.set_num_threads(1)

env = ZeldaEnv("rom/oos.gbc")
model = load_model(args.model)
state = read_state("states/horon-endgame.state")
steps = menu = dialog = swaps = deaths = 0
rooms, kinds, kills, seconds, acts = [], [], [], [], Counter()
for g in range(args.games):
    seen = []

    def on_step(e, t, buttons):
        global steps, menu, dialog
        steps += 1
        menu += bool(e.mem[ram.OPENED_MENU])
        dialog += bool(e.mem[ram.TEXT_ACTIVE])
        acts["+".join(sorted(buttons)) or "rien"] += 1
        seen.append((e.mem[EQUIP_B], e.mem[EQUIP_A]))

    t = solo(env, torch_policy(model, seed=g), state, args.seconds, render=True, on_step=on_step,
             warp=True, rng=random.Random(g))
    swaps += len(set(seen)) > 1
    deaths += t.dead
    rooms.append(len({r[:2] for r in t.rooms}))
    kinds.append(len(t.kinds))
    kills.append(t.kills)
    seconds.append(t.frames / 60)
res = {"games": args.games, "salles": sum(rooms) / args.games, "salles_max": max(rooms),
       "types": sum(kinds) / args.games, "kills": sum(kills) / args.games,
       "kills_par_type": sum(kills) / max(1, sum(kinds)), "morts_pct": 100 * deaths / args.games,
       "duree_s": sum(seconds) / args.games, "menu_pct": 100 * menu / max(1, steps),
       "dialogue_pct": 100 * dialog / max(1, steps), "items_changes": swaps,
       "actions_pct": {a: round(100 * n / max(1, steps), 1) for a, n in acts.most_common(8)}}
print(json.dumps(res, ensure_ascii=False))
if args.out:
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
