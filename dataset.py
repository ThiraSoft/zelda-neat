"""Dataset pour un world model : des parties enregistrées en (écran, boutons), un fichier par partie.

Chaque worker enchaîne des parties qui partent d'une cellule tirée au hasard (au plus 8 par salle,
pour que chaque salle pèse à peu près autant) dans l'archive (ou de Horon), jouées par le modèle PPO (échantillonné, graine différente à chaque partie) ou par des
actions aléatoires collantes. Avec plusieurs modèles, chaque partie jouée par un modèle le tire
au hasard parmi eux, à parts égales. Une partie est continue du début à la fin, les clips d'entraînement
se prennent à l'intérieur.

    dataset.py --archive spawns --out /mnt/data/datasets/zelda --max-gb 50 --model models/ppo.pt models/ppo-kills.pt

Fichier ep-<graine>-<worker>-<n>.npz : frames (N, 144, 160, 3) uint8 en RGB, écran après l'action,
actions (N,) uint8 (bits haut, bas, gauche, droite, A, B tenus FRAMES_PER_ACTION frames),
first (144, 160, 3) l'écran avant la première action, plus start (cellule ou "horon") et policy
(nom du modèle sans extension, ou "random").
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
    from zneat.env import EQUIP_A, EQUIP_B, ZeldaEnv, to_mask, tile_value
    from zneat.runner import FRAMES_PER_ACTION, random_policy

    models = {}
    if args.model:
        import torch
        torch.set_num_threads(1)
        from zneat.ppo_model import load_model, torch_policy
        for m in args.model:
            if os.path.exists(m):
                try:
                    name = os.path.splitext(os.path.basename(m))[0]
                    models[name] = load_model(m)
                except Exception as e:
                    print(f"worker {i}: impossible de charger {m}: {e}", flush=True)
            else:
                print(f"worker {i}: modèle {m} introuvable, ignoré", flush=True)
    rng = random.Random(args.seed * 1000 + i)
    env = ZeldaEnv(args.rom)
    steps = int(args.seconds * 60 / FRAMES_PER_ACTION)
    n = 0
    while True:
        if folder_bytes(args.out) >= args.max_gb * 1e9:
            return

        r = rng.random()
        if r < args.horon:
            start_label = "horon"
            env.load(read_state(args.start))
        elif r < args.horon + args.warp:
            # Téléportation vers une carte aléatoire du monde entier (Overworld, Subrosia, Donjons)
            env.load(read_state(args.start))
            g = rng.choice([0, 0, 1, 4, 5])
            room_id = rng.randint(0, 63 if g == 1 else 255)
            env.pb.memory[0xCC63] = 0x80 | g
            env.pb.memory[0xCC64] = room_id
            env.pb.memory[0xCC66] = 0x45
            env.pb.memory[0xCC67] = 0x01
            env.pb.memory[ram.SEASON] = rng.randint(0, 3)
            for _ in range(35):
                env.step(set(), 1, False)

            m = env.mem
            coll = m[ram.ROOM_COLLISIONS:ram.ROOM_COLLISIONS + ram.ROOM_STRIDE * ram.ROOM_ROWS]
            rows, cols = (11, 15) if m[ram.ROOM_IS_LARGE] else (8, 10)
            free = [(y, x) for y in range(1, rows - 1) for x in range(1, cols - 1)
                    if tile_value(coll[y * ram.ROOM_STRIDE + x]) == 0.0]
            if free:
                ty, tx = rng.choice(free)
                env.pb.memory[ram.LINK + ram.OBJ_YH] = (ty << 4) + 8
                env.pb.memory[ram.LINK + ram.OBJ_XH] = (tx << 4) + 8
            start_label = f"warp-{g}-{room_id}"
        else:
            cells = room_capped(args.archive, args.cells_per_room)
            if cells:
                cell_path = rng.choice(cells)
                start_label = os.path.basename(cell_path)
                env.load(read_state(cell_path))
            else:
                start_label = "horon"
                env.load(read_state(args.start))

        # Calibrage identique à l'entraînement : 4 coeurs (16 quarts), 0 rubis, épée sur B, bouclier sur A
        env.pb.memory[ram.LINK_MAX_HEALTH] = min(env.mem[ram.LINK_MAX_HEALTH], 16)
        env.pb.memory[ram.LINK_HEALTH] = min(env.mem[ram.LINK_HEALTH], 16)
        env.pb.memory[ram.RUPEES] = 0
        env.pb.memory[ram.RUPEES + 1] = 0
        env.pb.memory[EQUIP_B] = 0x05
        env.pb.memory[EQUIP_A] = 0x01
        env.pb.memory[0xCBEA] = 0x09
        for _ in range(15):
            env.step(set(), 1, False)

        if models and rng.random() < args.model_frac:
            name = rng.choice(sorted(models))
            policy = torch_policy(models[name], seed=rng.getrandbits(31))
        else:
            name = "random"
            policy = random_policy(random.Random(rng.getrandbits(32)))
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
        out = os.path.join(args.out, f"ep-{args.seed}-{i:02d}-{n:06d}.npz")
        np.savez_compressed(out + ".tmp.npz", frames=frames[:k], actions=actions[:k], first=first,
                            start=start_label, policy=name)
        os.replace(out + ".tmp.npz", out)
        n += 1


def main():
    default_archive = next((d for d in ["runs/horon/archive", "runs/goexplore/archive", "spawns"] if os.path.exists(d)), "spawns")
    default_candidates = [m for m in ["runs/checkpoints-20m/model-best.pt", "runs/ppo-impala/model-best.pt", "models/ppo.pt"] if os.path.exists(m)]
    default_models = [default_candidates[0]] if default_candidates else []

    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="rom/oos.gbc")
    ap.add_argument("--start", default="states/horon-endgame.state")
    ap.add_argument("--archive", default=default_archive, help="dossier de cellules *.state.z où partir (défaut auto)")
    ap.add_argument("--cells-per-room", type=int, default=8)
    ap.add_argument("--out", default="/mnt/data/datasets/zelda")
    ap.add_argument("--model", nargs="*", default=default_models,
                    help="modèles PPO (défaut : meilleur modèle trouvé) ; aucun = 100%% aléatoire")
    ap.add_argument("--model-frac", type=float, default=0.7, help="part des parties jouées par le modèle (70%% modèle, 30%% aléatoire)")
    ap.add_argument("--horon", type=float, default=0.1, help="part des parties qui partent de Horon")
    ap.add_argument("--warp", type=float, default=0.5,
                    help="part des parties téléportées sur des cartes aléatoires (Overworld, Subrosia, Donjons)")
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
