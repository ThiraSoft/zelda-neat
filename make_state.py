"""Joue dans la fenêtre jusqu'à l'endroit voulu, puis ferme-la : l'état est sauvegardé."""
import argparse

from pyboy import PyBoy

ap = argparse.ArgumentParser()
ap.add_argument("--rom", default="rom/oos.gbc")
ap.add_argument("--out", default="states/after-intro.state")
ap.add_argument("--load", help="partir d'un state existant")
args = ap.parse_args()

pb = PyBoy(args.rom, window="SDL2")
if args.load:
    with open(args.load, "rb") as f:
        pb.load_state(f)
while pb.tick():
    pass
with open(args.out, "wb") as f:
    pb.save_state(f)
pb.stop(save=False)
print("state écrit :", args.out)
