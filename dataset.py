"""Dataset pour un world model : des parties enregistrées en (écran, boutons), un fichier par partie.

Chaque worker enchaîne des parties qui partent d'une cellule tirée au hasard (au plus 8 par salle,
pour que chaque salle pèse à peu près autant) dans l'archive (ou de Horon), jouées par le modèle PPO (échantillonné, graine différente à chaque partie) ou par des
actions aléatoires collantes. Une partie est continue du début à la fin, les clips d'entraînement
se prennent à l'intérieur.

    dataset.py --archive spawns --out /mnt/data/datasets/zelda --max-gb 50 --model models/ppo.pt

Fichier ep-<graine>-<worker>-<n>.npz : frames (N, 144, 160, 3) uint8 en RGB, écran après l'action,
actions (N,) uint8 (bits haut, bas, gauche, droite, A, B tenus FRAMES_PER_ACTION frames),
first (144, 160, 3) l'écran avant la première action, plus start (cellule ou "horon") et policy.
"""
import argparse
import glob
import json
import os
import random
import signal
import time
from multiprocessing import Process

import numpy as np


def folder_bytes(path):
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(path, "*.npz")))


def room_capped(archive, per_room):
    """Au plus per_room cellules par salle (groupe, salle), tirées avec une graine fixe."""
    by_room = {}
    for c in sorted(glob.glob(os.path.join(archive, "*.state.z"))):
        by_room.setdefault(tuple(os.path.basename(c).split("-")[:2]), []).append(c)
    rng = random.Random(0)
    return [c for cs in by_room.values() for c in rng.sample(cs, min(len(cs), per_room))]


def worker(i, args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from zneat import ram
    from zneat.archive import read_state
    from zneat.env import ZeldaEnv, to_mask
    from zneat.runner import FRAMES_PER_ACTION, random_policy

    model = None
    if args.model:
        import torch
        torch.set_num_threads(1)
        from zneat.ppo_model import load_model, torch_policy
        model = load_model(args.model)
    rng = random.Random(args.seed * 1000 + i)
    env = ZeldaEnv(args.rom)
    steps = int(args.seconds * 60 / FRAMES_PER_ACTION)
    n = 0
    while True:
        if folder_bytes(args.out) >= args.max_gb * 1e9:
            return
        cells = room_capped(args.archive, args.cells_per_room)
        if cells and rng.random() >= args.horon:
            start = rng.choice(cells)
        else:
            start = args.start
        use_model = model is not None and rng.random() < args.model_frac
        policy = (torch_policy(model, seed=rng.getrandbits(31)) if use_model
                  else random_policy(random.Random(rng.getrandbits(32))))
        env.load(read_state(start))
        env.step(set(), 1, True)
        first = env.pb.screen.ndarray[:, :, :3].copy()
        frames = np.empty((steps, 144, 160, 3), np.uint8)
        actions = np.empty(steps, np.uint8)
        k = 0
        for k in range(steps):
            buttons = policy(env)
            env.step(buttons, FRAMES_PER_ACTION, True)
            frames[k] = env.pb.screen.ndarray[:, :, :3]
            actions[k] = to_mask(buttons)
            if env.mem[ram.LINK_HEALTH] == 0:   # mort : la partie s'arrête là
                k += 1
                break
        else:
            k = steps
        name = os.path.join(args.out, f"ep-{args.seed}-{i:02d}-{n:06d}.npz")
        np.savez_compressed(name + ".tmp.npz", frames=frames[:k], actions=actions[:k], first=first,
                            start=os.path.basename(start), policy="ppo" if use_model else "random")
        os.replace(name + ".tmp.npz", name)
        n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="rom/oos.gbc")
    ap.add_argument("--start", default="states/horon-endgame.state")
    ap.add_argument("--archive", default="spawns", help="dossier de cellules *.state.z où partir")
    ap.add_argument("--cells-per-room", type=int, default=8)
    ap.add_argument("--out", default="/mnt/data/datasets/zelda")
    ap.add_argument("--model", default="models/ppo.pt", help="modèle PPO (sans : actions aléatoires seulement)")
    ap.add_argument("--model-frac", type=float, default=0.7, help="part des parties jouées par le modèle")
    ap.add_argument("--horon", type=float, default=0.1, help="part des parties qui partent de Horon")
    ap.add_argument("--seconds", type=float, default=120, help="durée d'une partie")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-gb", type=float, default=50)
    ap.add_argument("--seed", type=int, default=int(time.time()))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, f"config-{args.seed}.json"), "w"), indent=1)
    procs = [Process(target=worker, args=(i, args)) for i in range(args.workers)]
    for p in procs:
        p.start()
    try:
        while any(p.is_alive() for p in procs):
            time.sleep(60)
            files = glob.glob(os.path.join(args.out, "ep-*.npz"))
            print(f"dataset : {len(files)} parties, {folder_bytes(args.out) / 1e9:.2f} Go "
                  f"sur {args.max_gb:.0f}", flush=True)
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
