"""Exploration Go-Explore par parties aléatoires, sans NEAT : à chaque tour, des cellules de
l'archive sont tirées et des parties aux actions aléatoires (collantes) en partent. Chaque cellule
nouvelle rejoint l'archive avec son state et le chemin d'inputs qui y mène.

    explore.py --run runs/horon --states states/horon-endgame.state --workers 3
"""
import argparse
import glob
import json
import os
import random
import time
from multiprocessing import Pool

from train import seed_archive
from zneat.archive import Archive
from zneat.runner import indexed_job, init_worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="rom/oos.gbc")
    ap.add_argument("--states", default="states/horon-endgame.state", help="graines de l'archive")
    ap.add_argument("--run", default="runs/horon")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--trials", type=int, default=None, help="parties par tour (défaut 50 par worker)")
    ap.add_argument("--seconds", type=float, default=20, help="durée max d'une partie")
    ap.add_argument("--patience", type=float, default=15, help="arrêt après N s sans rien de nouveau")
    args = ap.parse_args()

    archive = Archive(os.path.join(args.run, "archive"))
    if not os.path.exists(archive.known_path):
        archive.save()
    if not len(archive):
        states = sorted(glob.glob(args.states))
        if not states:
            raise SystemExit(f"aucun save state dans {args.states}")
        seed_archive(archive, args.rom, states)
    kills_path = os.path.join(args.run, "kills.json")
    kills = json.load(open(kills_path)) if os.path.exists(kills_path) else {"random": 0, "kinds": []}
    trials = args.trials or 50 * args.workers
    rng = random.Random()
    pool = Pool(args.workers, initializer=init_worker,
                initargs=(args.rom, (), args.seconds, args.patience))
    turn = max([c["generation"] for c in archive.cells.values()] + [-1]) + 1
    try:
        while True:
            t0 = time.time()
            jobs = [("random", archive.known_path, (k, archive.path(k)), rng.getrandbits(32),
                     int(args.seconds * 60)) for k in archive.choose(trials, rng)]
            found, gen_kills, kinds = 0, 0, set(kills["kinds"])
            for _, (_, captured, n_kills, killed) in pool.imap_unordered(indexed_job, enumerate(jobs)):
                gen_kills += n_kills
                kinds |= killed
                for k, (z, parent, acts) in captured.items():
                    found += archive.add(k, z, turn, "random", parent, acts)
            archive.save()
            kills["random"] = kills.get("random", 0) + gen_kills
            kills["kinds"] = sorted(kinds)
            json.dump(kills, open(kills_path, "w"))
            top = max(k[3] for k in archive.cells)
            print(f"tour {turn} : {time.time() - t0:.0f} s")
            print(f"archive : {len(archive)} cellules, {len(archive.rooms())} salles, "
                  f"{len({r[:2] for r in archive.rooms()})} sans saison, histoire max {top} (+{found})")
            print(f"kills : {gen_kills} ce tour, {kills['random']} au total, "
                  f"{len(kinds)} types {[hex(k) for k in sorted(kinds)]}")
            turn += 1
    except KeyboardInterrupt:
        pool.terminate()
        archive.save()


if __name__ == "__main__":
    main()
