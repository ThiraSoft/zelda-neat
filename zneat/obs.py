"""Observation et actions du joueur PPO : écran en niveaux de gris, grille de NEAT et repères
(salle, position, salles voisines déjà vues). Sans torch."""
import numpy as np

from . import ram
from .env import NUM_INPUTS, to_mask

DIRS = ("up", "down", "left", "right")
ACTIONS = ([set()] + [{d} for d in DIRS] + [{"a"}, {"b"}]
           + [{d, "b"} for d in DIRS] + [{d, "a"} for d in DIRS])

PPO_INPUTS = NUM_INPUTS + 8 + 2 + 8 + 2 + 4


class Observer:
    """Observation du PPO : celle de NEAT (env.observe) plus de quoi savoir où on est.
    Groupe (one-hot 8), salle en coordonnées (y, x sur la grille 16x16 du groupe) et en 8 bits,
    position de Link dans la salle, et pour les 4 salles voisines : déjà vue pendant l'épisode ?
    Un Observer par épisode, il retient les salles vues."""

    def __init__(self, visited=()):
        self.visited = set(visited)

    def __call__(self, env):
        m = env.mem
        group, room = m[ram.ACTIVE_GROUP], m[ram.ACTIVE_ROOM]
        self.visited.add((group, room))
        rows, cols = (11, 15) if m[ram.ROOM_IS_LARGE] else (8, 10)
        x = np.zeros(PPO_INPUTS - NUM_INPUTS, dtype=np.float32)
        x[min(group, 7)] = 1.0
        x[8], x[9] = (room >> 4) / 15, (room & 15) / 15
        x[10:18] = [(room >> i) & 1 for i in range(8)]
        x[18] = min(1.0, m[ram.LINK + ram.OBJ_YH] / (16 * rows))
        x[19] = min(1.0, m[ram.LINK + ram.OBJ_XH] / (16 * cols))
        for i, d in enumerate((-16, 16, -1, 1)):   # haut, bas, gauche, droite
            n = room + d
            if 0 <= n < 256 and (d in (-16, 16) or n >> 4 == room >> 4):
                x[20 + i] = 1.0 if (group, n) in self.visited else 0.0
        return np.concatenate((env.observe(), x))


def observe(env, observer):
    """(écran 72x80 uint8, vecteur PPO_INPUTS float32). L'écran doit avoir été rendu au dernier pas."""
    return env.frame(), observer(env)


MASK_TO_ACTION = {to_mask(a): i for i, a in enumerate(ACTIONS)}

