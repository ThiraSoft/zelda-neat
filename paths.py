"""Chemins d'inputs de l'archive Go-Explore.

    paths.py --archive runs/horon/archive              # combien de cellules joignables depuis Horon
    paths.py --archive runs/horon/archive --check 20   # rejoue 20 chemins et vérifie l'arrivée
"""
import argparse
import random
from multiprocessing import Pool

from zneat import ram
from zneat.archive import Archive, key_str, read_state
from zneat.env import ZeldaEnv
from zneat.paths import pick_targets, replay, seed_state

_env = None


def init(rom):
    global _env
    _env = ZeldaEnv(rom)


def check(item):
    root, key = item
    archive = Archive(root)
    seed, actions = archive.full_path(key)
    got = replay(_env, seed_state(archive, seed), actions)
    # Plus fin que la cellule : position au pixel près, PV et salle doivent coller au state archivé.
    probe = (ram.LINK + ram.OBJ_YH, ram.LINK + ram.OBJ_YH - 1, ram.LINK + ram.OBJ_XH,
             ram.LINK + ram.OBJ_XH - 1, ram.LINK_HEALTH, ram.ACTIVE_ROOM)
    mine = [_env.mem[a] for a in probe]
    _env.load(read_state(archive.path(key)))
    ref = [_env.mem[a] for a in probe]
    return key, got if mine == ref else (got, mine, ref), len(actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="rom/oos.gbc")
    ap.add_argument("--archive", default="runs/horon/archive")
    ap.add_argument("--check", type=int, default=0, help="rejoue N chemins tirés au hasard")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    archive = Archive(args.archive)
    reach = archive.with_paths()
    print(f"{len(archive)} cellules, {len(reach)} joignables depuis la graine par un chemin enregistré")
    if not reach:
        print("aucun chemin : l'archive date d'avant l'enregistrement des chemins, il faut un nouveau run")
        return
    longest = max(reach.values())
    print(f"chemin le plus long : {longest} pas ({longest * 4 / 60:.0f} s de jeu)")
    for k in pick_targets(archive, 5):
        print(f"  cible {key_str(k)} : histoire {k[3]}, {reach[k]} pas")
    if args.check:
        keys = random.sample(sorted(reach), min(args.check, len(reach)))
        ok = 0
        with Pool(args.workers, initializer=init, initargs=(args.rom,)) as pool:
            for key, got, n in pool.imap_unordered(check, [(args.archive, k) for k in keys]):
                good = got == key
                ok += good
                if not good:
                    print(f"  ÉCART {key_str(key)} : arrivé en {key_str(got)} après {n} pas")
        print(f"vérification : {ok}/{len(keys)} chemins arrivent dans la bonne cellule")


if __name__ == "__main__":
    main()
