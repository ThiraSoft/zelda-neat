"""Chemins d'inputs de l'archive : rejouer depuis la graine (Horon) jusqu'à une cellule,
et en tirer une démo (states réguliers le long du chemin) pour le PPO."""
import json
import os
import zlib

import numpy as np

from . import ram
from .archive import cell_key, key_str, read_state
from .env import SCREEN, from_mask
from .fitness import Tracker
from .obs import MASK_TO_ACTION, PPO_INPUTS, Observer, observe
from .runner import FRAMES_PER_ACTION


def replay(env, seed_state, actions):
    """Rejoue `actions` depuis `seed_state` et renvoie la cellule d'arrivée."""
    env.load(seed_state)
    for a in actions:
        env.step(from_mask(a), FRAMES_PER_ACTION)
    return cell_key(env.mem)


def build_demo(env, seed_state, actions, out_dir, every=32):
    """Rejoue le chemin et sauve un state tous les `every` pas dans out_dir/<i>.state.z.
    meta.json donne pour chaque state : le pas, les salles (groupe, salle) déjà vues,
    les types d'ennemis tués et les bits d'histoire gagnés depuis le départ.
    Pour l'imitation, frames.npy, vec.npy et act.npy gardent l'observation avant chaque pas et
    l'action jouée (-1 si elle n'est pas dans obs.ACTIONS)."""
    os.makedirs(out_dir, exist_ok=True)
    env.load(seed_state)
    t = Tracker(env.mem)
    observer = Observer()
    frames = np.empty((len(actions),) + SCREEN, np.uint8)
    vecs = np.empty((len(actions), PPO_INPUTS), np.float16)
    acts = np.array([MASK_TO_ACTION.get(a, -1) for a in actions], np.int8)
    first_seen = {}     # salle -> premier index de state où elle est vue
    points = []

    def snap(step):
        i = len(points)
        for r in t.rooms:
            first_seen.setdefault(f"{r[0]}-{r[1]}", i)
        with open(os.path.join(out_dir, f"{i}.state.z"), "wb") as f:
            f.write(zlib.compress(env.save(), 1))
        points.append({"step": step, "rooms": len({r[:2] for r in t.rooms}),
                       "kinds": sorted(t.kinds), "story": t.story})

    snap(0)
    for n, a in enumerate(actions, 1):
        frames[n - 1], vecs[n - 1] = observe(env, observer)
        env.step(from_mask(a), FRAMES_PER_ACTION, True)
        t.update(env.mem, FRAMES_PER_ACTION)
        if n % every == 0 or n == len(actions):
            snap(n)
    meta = {"points": points, "first_seen": first_seen, "end": key_str(cell_key(env.mem)),
            "dead": t.dead, "health": env.mem[ram.LINK_HEALTH]}
    np.save(os.path.join(out_dir, "frames.npy"), frames)
    np.save(os.path.join(out_dir, "vec.npy"), vecs)
    np.save(os.path.join(out_dir, "act.npy"), acts)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def pick_targets(archive, n):
    """Les n cellules joignables depuis la graine les plus avancées : histoire d'abord,
    puis salles distinctes de la chaîne, puis chemin le plus court. Une seule par salle."""
    reach = archive.with_paths()
    scored = []
    for k, steps in reach.items():
        rooms = {c[:2] for c in archive.chain(k)}
        scored.append(((k[3], len(rooms), -steps), k))
    scored.sort(reverse=True)
    out, seen = [], set()
    for _, k in scored:
        if k[:3] not in seen:
            seen.add(k[:3])
            out.append(k)
        if len(out) == n:
            break
    return out


def seed_state(archive, key):
    return read_state(archive.path(key))
