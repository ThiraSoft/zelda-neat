import json
import math
import os
import pickle
import sys
import zlib

from . import ram
from .fitness import story_bits


def cell_key(m):
    """Cellule Go-Explore : (groupe, salle, saison, bits d'histoire, zone y, zone x).
    Les bits d'histoire font d'une même salle "après l'événement" une cellule nouvelle,
    la zone 3x3 garde aussi la progression à l'intérieur d'une grande salle."""
    rows, cols = (11, 15) if m[ram.ROOM_IS_LARGE] else (8, 10)
    zy = min(2, (m[ram.LINK + ram.OBJ_YH] >> 4) * 3 // rows)
    zx = min(2, (m[ram.LINK + ram.OBJ_XH] >> 4) * 3 // cols)
    return (m[ram.ACTIVE_GROUP], m[ram.ACTIVE_ROOM], m[ram.SEASON], story_bits(m), zy, zx)


def key_str(k):
    return "-".join(str(x) for x in k)


def read_state(path):
    with open(path, "rb") as f:
        data = f.read()
    return zlib.decompress(data) if path.endswith(".z") else data


class Archive:
    """Un save state compressé par cellule, plus index.json et known.pkl pour les workers."""

    def __init__(self, root):
        self.root = root
        self.index_path = os.path.join(root, "index.json")
        self.known_path = os.path.join(root, "known.pkl")
        os.makedirs(root, exist_ok=True)
        self.cells = {}
        if os.path.exists(self.index_path):
            for k, v in json.load(open(self.index_path)).items():
                self.cells[tuple(int(x) for x in k.split("-"))] = v

    def __len__(self):
        return len(self.cells)

    def keys(self):
        return frozenset(self.cells)

    def rooms(self):
        return {k[:3] for k in self.cells}

    def path(self, key):
        return os.path.join(self.root, self.cells[key]["file"])

    def add(self, key, zstate, generation, source, parent=None, actions=None):
        """`parent` : cellule d'où la partie est partie, `actions` : inputs joués depuis son state
        jusqu'à celui-ci (un octet par pas, voir env.to_mask), dans <clé>.path.z."""
        if key in self.cells:
            return False
        name = key_str(key) + ".state.z"
        with open(os.path.join(self.root, name), "wb") as f:
            f.write(zstate)
        self.cells[key] = {"file": name, "generation": generation, "source": source, "chosen": 0}
        if parent is not None:
            with open(os.path.join(self.root, key_str(key) + ".path.z"), "wb") as f:
                f.write(zlib.compress(actions))
            self.cells[key].update(parent=key_str(parent), steps=len(actions))
        return True

    def chain(self, key):
        """Cellules de la graine jusqu'à `key`, ou None si un maillon n'a pas de chemin enregistré
        (cellules d'avant l'enregistrement des chemins)."""
        out = [key]
        while self.cells[out[-1]]["source"] != "seed":
            parent = self.cells[out[-1]].get("parent")
            if parent is None:
                return None
            out.append(tuple(int(x) for x in parent.split("-")))
        return out[::-1]

    def segment(self, key):
        with open(os.path.join(self.root, key_str(key) + ".path.z"), "rb") as f:
            return zlib.decompress(f.read())

    def full_path(self, key):
        """(graine, inputs depuis le state de la graine jusqu'à `key`) ou None."""
        chain = self.chain(key)
        if chain is None:
            return None
        return chain[0], b"".join(self.segment(k) for k in chain[1:])

    def with_paths(self):
        """Cellules qu'on sait rejoindre depuis une graine, avec la longueur du chemin en pas."""
        out, steps = {}, {}

        def length(k):
            if k not in steps:
                c = self.cells[k]
                if c["source"] == "seed":
                    steps[k] = 0
                elif "parent" not in c:
                    steps[k] = None
                else:
                    p = tuple(int(x) for x in c["parent"].split("-"))
                    steps[k] = None  # garde contre les cycles
                    base = length(p)
                    steps[k] = None if base is None else base + c["steps"]
            return steps[k]

        sys.setrecursionlimit(max(10000, 4 * len(self.cells)))
        for k in self.cells:
            if length(k) is not None and self.cells[k]["source"] != "seed":
                out[k] = steps[k]
        return out

    def choose(self, n, rng):
        """Tirage avec remise. Favorise les cellules peu choisies et celles qui sont plus loin
        dans l'histoire, car c'est là qu'est la frontière."""
        keys = list(self.cells)
        top = max(k[3] for k in keys)
        w = [(1 + 2 * (k[3] == top)) / math.sqrt(1 + self.cells[k]["chosen"]) for k in keys]
        picked = rng.choices(keys, w, k=n)
        for k in picked:
            self.cells[k]["chosen"] += 1
        return picked

    def save(self):
        tmp = self.index_path + ".tmp"
        json.dump({key_str(k): v for k, v in self.cells.items()}, open(tmp, "w"))
        os.replace(tmp, self.index_path)
        with open(self.known_path + ".tmp", "wb") as f:
            pickle.dump(self.keys(), f)
        os.replace(self.known_path + ".tmp", self.known_path)
