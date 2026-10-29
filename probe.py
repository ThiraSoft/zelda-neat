"""Joue à la main et affiche ce que l'IA lit en RAM, pour vérifier les adresses."""
import argparse

from zneat import ram
from zneat.env import VIEW, ZeldaEnv
from zneat.fitness import Tracker

ap = argparse.ArgumentParser()
ap.add_argument("--rom", default="rom/oos.gbc")
ap.add_argument("--load", default="states/after-intro.state")
args = ap.parse_args()

env = ZeldaEnv(args.rom, window="SDL2", speed=1)
env.load(open(args.load, "rb").read())
t = Tracker(env.mem)
frame = 0
while env.pb.tick():
    frame += 1
    t.update(env.mem, 1)
    if frame % 60 == 0:
        m = env.mem
        obs = env.observe()
        print(f"salle={t.last_room} pv={m[ram.LINK_HEALTH]}/{m[ram.LINK_MAX_HEALTH]} "
              f"pos=({m[ram.LINK + ram.OBJ_YH]},{m[ram.LINK + ram.OBJ_XH]}) "
              f"salles={len(t.rooms)} cases={len(t.tiles)} mobs={sorted(t.kinds)} story={t.story} fit={t.fitness():.1f}")
        for row in obs[:VIEW * VIEW].reshape(VIEW, VIEW):
            print("".join({1.0: "#", 0.5: "~", 0.0: ".", -0.5: "P", -1.0: "E"}[v] for v in row))
