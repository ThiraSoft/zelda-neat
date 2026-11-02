"""Dataset pour un world model : des parties enregistrées en (écran, boutons), un fichier par partie.

Chaque worker joue --envs parties en même temps et enchaîne des parties qui partent d'une cellule tirée au hasard (au plus 8 par salle,
pour que chaque salle pèse à peu près autant) dans l'archive (ou de Horon), jouées par le modèle PPO (échantillonné, graine différente à chaque partie) ou par des
actions aléatoires collantes. Avec plusieurs modèles, chaque partie jouée par un modèle le tire
au hasard parmi eux, à parts égales. Une partie est continue du début à la fin, les clips d'entraînement
se prennent à l'intérieur.

    dataset.py --archive spawns --out /mnt/data/datasets/zelda --max-gb 50 --model models/ppo.pt

Fichier ep-<graine>-<worker>-<n>.npz : frames (N, 144, 160, 3) uint8 en RGB, écran après l'action,
actions (N,) uint8 (bits haut, bas, gauche, droite, A, B, Start tenus FRAMES_PER_ACTION frames),
scripted (N,) bool (pas joués par la séquence de menu, pas par la politique),
first (144, 160, 3) l'écran avant la première action, plus start (cellule ou "horon"), policy
(nom du modèle sans extension, ou "random") et frames_per_action (absent des anciens fichiers, à 4 frames).
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


MENU_OPEN_STEPS = 4    # pas d'attente après Start pour que le menu s'ouvre
MENU_MAX_MOVES = 4     # déplacements du curseur au plus
MENU_EQUIP_TRIES = 4   # essais pour équiper un vrai item
MENU_CLOSE_STEPS = 4   # pas d'attente après Start pour que le menu se ferme
MENU_CLOSE_TRIES = 3   # appuis sur Start au plus pour le refermer
MENU_MEAN_STEPS = 23   # longueur moyenne mesurée d'une séquence


def menu_start_chance(frac):
    """Probabilité par pas de lancer une séquence pour que frac des pas soient scriptés."""
    return frac / ((1 - frac) * MENU_MEAN_STEPS) if frac > 0 else 0.0


def can_open_menu(m, ram):
    return not (m[ram.TEXT_ACTIVE] or m[ram.OPENED_MENU] or m[ram.DISABLED_OBJECTS] & 0x81)


def menu_script(env, rng, ram):
    """Générateur de boutons : ouvre le menu, déplace le curseur au hasard, équipe l'item sous le curseur
    sur A ou B, referme. Abandonne si le menu ne s'ouvre pas (transition de salle, par exemple)."""
    from zneat.env import EQUIP_A, EQUIP_B
    yield {"start"}
    for _ in range(MENU_OPEN_STEPS):
        yield set()
    if not env.mem[ram.OPENED_MENU]:
        return
    # Le curseur peut être sur une case vide, ce qui laisserait le bouton sans item : on réessaie ailleurs.
    for _ in range(MENU_EQUIP_TRIES):
        for _ in range(rng.randint(1, MENU_MAX_MOVES)):
            yield {rng.choice(("up", "down", "left", "right"))}
            yield set()
        button = rng.choice(("a", "b"))
        yield {button}
        yield set()
        if env.mem[EQUIP_A if button == "a" else EQUIP_B]:
            break
    for _ in range(MENU_CLOSE_TRIES):
        if not env.mem[ram.OPENED_MENU]:
            return
        yield {"start"}
        for _ in range(MENU_CLOSE_STEPS):
            yield set()


def worker(i, args):
    """args.envs parties en parallèle dans ce processus, au même rythme : à chaque pas, toutes les parties
    jouées par un même modèle passent dans le réseau en un seul lot."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from zneat import ram
    from zneat.archive import read_state
    from zneat.env import EQUIP_A, EQUIP_B, ZeldaEnv, to_mask, warp_random
    from zneat.runner import FRAMES_PER_ACTION, random_policy

    rng = random.Random(args.seed * 1000 + i)
    models = {}
    if args.model:
        import torch
        torch.set_num_threads(1)
        from zneat.ppo_model import BatchPolicy, load_model
        for m in args.model:
            if os.path.exists(m):
                try:
                    models[os.path.splitext(os.path.basename(m))[0]] = BatchPolicy(load_model(m), rng.getrandbits(31))
                except Exception as e:
                    print(f"worker {i}: impossible de charger {m}: {e}", flush=True)
            else:
                print(f"worker {i}: modèle {m} introuvable, ignoré", flush=True)
    steps = int(args.seconds * 60 / FRAMES_PER_ACTION)
    menu_chance = menu_start_chance(args.menu_frac)
    counter = [0]

    def new_game(env):
        r = rng.random()
        if r < args.horon:
            start_label = "horon"
            env.load(read_state(args.start))
        elif r < args.horon + args.warp:
            # Téléportation vers une carte aléatoire du monde entier (Overworld, Subrosia, Donjons)
            env.load(read_state(args.start))
            warp_random(env, rng)
            start_label = f"warp-{env.mem[ram.ACTIVE_GROUP]}-{env.mem[ram.ACTIVE_ROOM]}"
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

        g = {"env": env, "start": start_label, "k": 0, "script": None,
             "frames": np.empty((steps, 144, 160, 3), np.uint8), "actions": np.empty(steps, np.uint8),
             "scripted": np.zeros(steps, bool)}
        if models and rng.random() < args.model_frac:
            g["policy"] = rng.choice(sorted(models))
            g["model"] = models[g["policy"]].new_game()
        else:
            g["policy"] = "random"
            g["random"] = random_policy(random.Random(rng.getrandbits(32)))
        env.step(set(), 1, True)
        g["first"] = env.pb.screen.ndarray[:, :, :3].copy()
        return g

    def save(g):
        k = g["k"]
        out = os.path.join(args.out, f"ep-{args.seed}-{i:02d}-{counter[0]:06d}.npz")
        counter[0] += 1
        np.savez_compressed(out + ".tmp.npz", frames=g["frames"][:k], actions=g["actions"][:k],
                            scripted=g["scripted"][:k], first=g["first"], start=g["start"], policy=g["policy"],
                            frames_per_action=FRAMES_PER_ACTION)
        os.replace(out + ".tmp.npz", out)

    if folder_bytes(args.out) >= args.max_gb * 1e9:
        return
    games = [new_game(ZeldaEnv(args.rom)) for _ in range(args.envs)]
    while games:
        buttons = [None] * len(games)
        for j, g in enumerate(games):
            env = g["env"]
            if g["script"] is None and rng.random() < menu_chance and can_open_menu(env.mem, ram):
                g["script"] = menu_script(env, rng, ram)
            if g["script"] is not None:
                buttons[j] = next(g["script"], None)
                if buttons[j] is None:
                    g["script"] = None
                else:
                    g["scripted"][g["k"]] = True
            if buttons[j] is None and g["policy"] == "random":
                buttons[j] = g["random"](env)
        for name, policy in models.items():
            batch = [j for j, g in enumerate(games) if buttons[j] is None and g["policy"] == name]
            if batch:
                for j, b in zip(batch, policy.act([games[j]["env"] for j in batch], [games[j]["model"] for j in batch])):
                    buttons[j] = b
        full = False
        for j, g in enumerate(games):
            env = g["env"]
            env.step(buttons[j], FRAMES_PER_ACTION, True)
            g["frames"][g["k"]] = env.pb.screen.ndarray[:, :, :3]
            g["actions"][g["k"]] = to_mask(buttons[j])
            g["k"] += 1
            if g["k"] >= steps or env.mem[ram.LINK_HEALTH] == 0:   # fin, ou mort : la partie s'arrête là
                save(g)
                full = full or folder_bytes(args.out) >= args.max_gb * 1e9
                games[j] = None if full else new_game(env)
        games = [g for g in games if g is not None]


def main():
    default_archive = next((d for d in ["runs/horon/archive", "runs/goexplore/archive", "spawns"] if os.path.exists(d)), "spawns")
    default_candidates = [m for m in ["models/ppo.pt"] if os.path.exists(m)]
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
    ap.add_argument("--menu-frac", type=float, default=0.05,
                    help="part des pas joués par la séquence de menu scriptée (changement d'item), 0 = aucune")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--envs", type=int, default=8, help="parties jouées en même temps par worker (réseau en lot)")
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
