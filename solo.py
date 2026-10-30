"""Partie solo : un génome NEAT ou un modèle PPO part de Horon et joue seul, sans save state.
Score = salles (groupe, salle) visitées + types d'ennemis tués.

    solo.py --genome runs/horon/best.pkl
    solo.py --model runs/ppo/model.pt [--greedy] [--window]
    solo.py --model models/ppo.pt --random-location [spawns]   # départ dans une cellule au hasard
"""
import argparse
import random

from zneat.archive import read_state
from zneat.env import ZeldaEnv
from zneat.runner import solo, solo_line

ap = argparse.ArgumentParser()
ap.add_argument("--rom", default="rom/oos.gbc")
ap.add_argument("--state", default="states/horon-endgame.state")
ap.add_argument("--config", default="config-neat.ini")
ap.add_argument("--genome", help="génome NEAT (.pkl)")
ap.add_argument("--model", help="modèle PPO (.pt)")
ap.add_argument("--greedy", action="store_true", help="PPO : action la plus probable au lieu d'échantillonner")
ap.add_argument("--seed", type=int, default=0, help="PPO : graine de l'échantillonnage")
ap.add_argument("--seconds", type=float, default=1200)
ap.add_argument("--window", action="store_true", help="affiche le jeu (vitesse réelle)")
ap.add_argument("--random-location", nargs="?", const="spawns", metavar="DOSSIER",
                help="part d'une cellule tirée au hasard dans DOSSIER (défaut spawns), comme dataset.py")
args = ap.parse_args()
if args.random_location:
    import glob
    import os
    from dataset import room_capped
    folder = args.random_location
    if not os.path.exists(folder) or not glob.glob(os.path.join(folder, "*.state*")):
        folder = "runs/horon/archive"
    candidates = room_capped(folder, 8)
    if candidates:
        args.state = random.Random(args.seed).choice(candidates)
        print("départ :", args.state, flush=True)

# La fenêtre SDL s'ouvre avant de charger torch : dans l'autre ordre, elle reste noire (ROCm).
env = ZeldaEnv(args.rom, window="SDL2" if args.window else "null", speed=1 if args.window else 0)

if args.model:
    from zneat.ppo_model import load_model, torch_policy
    policy = torch_policy(load_model(args.model), args.greedy, args.seed)
elif args.genome:
    import pickle

    import neat

    from zneat.runner import make_net, net_policy
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)
    policy = net_policy(make_net(pickle.load(open(args.genome, "rb")), config))
else:
    raise SystemExit("--genome ou --model")

# Le modèle PPO regarde l'écran : il faut le rendre à chaque pas.
t = solo(env, policy, read_state(args.state), args.seconds, render=args.window or bool(args.model))
print("solo :", solo_line(t))
