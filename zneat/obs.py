"""Observation et actions du joueur PPO : écran en niveaux de gris, grille de NEAT et repères
(salle, position, salles voisines déjà vues). Sans torch."""
import numpy as np

from . import ram
from .env import NUM_INPUTS, ROOM_GRID, to_mask

DIRS = ("up", "down", "left", "right")
ACTIONS = ([set()] + [{d} for d in DIRS] + [{"a"}, {"b"}]
           + [{d, "b"} for d in DIRS] + [{d, "a"} for d in DIRS]
           + [{"start"}])

# Périodes pour l'encodage temporel sinusoïdal (en pas d'action, 5 pas = 1s) :
# 5 pas (1s : rythme de frappe/esquive), 20 pas (4s : cycle d'attaques des ennemis/invincibilité),
# 100 pas (20s : moyen terme), 500 pas (100s : échelle de l'épisode).
TIME_PERIODS = (5, 20, 100, 500)
TIME_INPUTS = 2 * len(TIME_PERIODS)   # sin et cos pour chaque période (8 valeurs)
MARKS = 8 + 2 + 8 + 2 + 4 + TIME_INPUTS
PPO_INPUTS = NUM_INPUTS + MARKS + ROOM_GRID


def room_marks(m, visited, out):
    """Position de Link dans la salle (2) puis, pour les 4 salles voisines (haut, bas, gauche, droite),
    1 si déjà vue pendant l'épisode (4). Écrit dans out[0:6]."""
    group, room = m[ram.ACTIVE_GROUP], m[ram.ACTIVE_ROOM]
    rows, cols = (11, 15) if m[ram.ROOM_IS_LARGE] else (8, 10)
    out[0] = min(1.0, m[ram.LINK + ram.OBJ_YH] / (16 * rows))
    out[1] = min(1.0, m[ram.LINK + ram.OBJ_XH] / (16 * cols))
    for i, d in enumerate((-16, 16, -1, 1)):
        n = room + d
        if 0 <= n < 256 and (d in (-16, 16) or n >> 4 == room >> 4):
            out[2 + i] = 1.0 if (group, n) in visited else 0.0


# Entrées de NEAT : celles d'env.observe, puis room_marks et le temps passé dans la salle (1 = 10 s).
NEAT_INPUTS = NUM_INPUTS + 7
NEAT_ROOM_STEPS = 50


class NeatObserver:
    """Mémoire minimale d'un génome NEAT (réseau sans état) : salles vues et temps dans la salle.
    Un par partie, voir new_game."""

    def __init__(self):
        self.new_game()

    def new_game(self):
        self.visited, self.room, self.steps = set(), None, 0

    def __call__(self, env):
        m = env.mem
        room = (m[ram.ACTIVE_GROUP], m[ram.ACTIVE_ROOM])
        if room != self.room:
            self.room, self.steps = room, 0
        else:
            self.steps += 1
        self.visited.add(room)
        x = np.zeros(7, dtype=np.float32)
        room_marks(m, self.visited, x)
        x[6] = min(1.0, self.steps / NEAT_ROOM_STEPS)
        return np.concatenate((env.observe(), x))


class Observer:
    """Observation du PPO : celle de NEAT (env.observe) plus de quoi savoir où on est.
    Groupe (one-hot 8), salle en coordonnées (y, x sur la grille 16x16 du groupe) et en 8 bits,
    position de Link dans la salle, pour les 4 salles voisines : déjà vue pendant l'épisode ?,
    encodage sinusoïdal temporel (sin et cos sur plusieurs échelles de temps), puis toute la salle (env.room_grid).
    Un Observer par épisode, il retient les salles vues et le pas courant."""

    def __init__(self, visited=(), step=0):
        self.visited = set(visited)
        self.step = step

    def __call__(self, env):
        m = env.mem
        group, room = m[ram.ACTIVE_GROUP], m[ram.ACTIVE_ROOM]
        self.visited.add((group, room))
        x = np.zeros(MARKS, dtype=np.float32)
        x[min(group, 7)] = 1.0
        x[8], x[9] = (room >> 4) / 15, (room & 15) / 15
        x[10:18] = [(room >> i) & 1 for i in range(8)]
        room_marks(m, self.visited, x[18:24])
        t = self.step
        for i, p in enumerate(TIME_PERIODS):
            w = 2.0 * np.pi * t / p
            x[24 + 2 * i] = np.sin(w)
            x[24 + 2 * i + 1] = np.cos(w)
        self.step += 1
        return np.concatenate((env.observe(), x, env.room_grid()))


def observe(env, observer):
    """(écran 72x80 uint8, vecteur PPO_INPUTS float32). L'écran doit avoir été rendu au dernier pas."""
    return env.frame(), observer(env)


MASK_TO_ACTION = {to_mask(a): i for i, a in enumerate(ACTIONS)}

