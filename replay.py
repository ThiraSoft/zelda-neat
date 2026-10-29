"""Regarde jouer le meilleur génome."""
import argparse
import glob
import pickle

import neat

from zneat.archive import read_state
from zneat.env import ZeldaEnv
from zneat.runner import make_net, play

ap = argparse.ArgumentParser()
ap.add_argument("--rom", default="rom/oos.gbc")
ap.add_argument("--states", default="states/*.state", help="ex. runs/horon/archive/*.state.z")
ap.add_argument("--config", default="config-neat.ini")
ap.add_argument("--genome", default="runs/horon/best.pkl")
ap.add_argument("--speed", type=float, default=1)
ap.add_argument("--max-seconds", type=float, default=120)
ap.add_argument("--patience", type=float, default=15)
args = ap.parse_args()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)
genome = pickle.load(open(args.genome, "rb"))
env = ZeldaEnv(args.rom, window="SDL2", speed=args.speed)
for path in sorted(glob.glob(args.states)):
    t = play(env, make_net(genome, config), read_state(path), render=True,
             max_frames=int(args.max_seconds * 60), patience=int(args.patience * 60))
    print(f"{path}: fitness={t.fitness():.1f} salles={len(t.rooms)} cases={len(t.tiles)} mobs={len(t.kinds)} story={t.story}")
