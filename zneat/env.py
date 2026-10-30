import io
import os

import numpy as np
from pyboy import PyBoy

from . import ram

BUTTONS = ("up", "down", "left", "right", "a", "b", "start")


def to_mask(buttons):
    return sum(1 << i for i, b in enumerate(BUTTONS) if b in buttons)


def from_mask(mask):
    return {b for i, b in enumerate(BUTTONS) if mask >> i & 1}
VIEW_RADIUS = 4
VIEW = 2 * VIEW_RADIUS + 1
NUM_INPUTS = VIEW * VIEW + 1 + 1 + 4 + 4


SCREEN = (72, 80)
LUMA = np.array([77, 151, 28], dtype=np.uint16)

SOUND_WAIT_JUMPS = (0x49D1, 0x49D2, 0x49EF, 0x49F0)

# Objet forcé sur A à chaque chargement (variable ZELDA_A, make A_ITEM=...). Les states de l'archive
# ont le Rod sur A ; "feather" y met la cape de Roc. Trésors : constants/common/treasure.s.
ITEMS = {"rod": 0x07, "feather": 0x17}
EQUIP_B, EQUIP_A, STORAGE = 0xC680, 0xC681, range(0xC682, 0xC692)


def tile_value(c):
    # Valeurs de collision : constants/common/specialCollisionValues.s
    if c < 0x10:
        return 1.0 if c else 0.0
    if c == 0x10:
        return 0.5
    if c >= 0xFE:
        return 1.0
    return 0.0  # ponts, escaliers, rails : praticables


class Ram:
    """Lit $d000-$dfff toujours dans la banque WRAM 1 (les objets), quelle que soit SVBK."""

    def __init__(self, mem):
        self.mem = mem

    def __getitem__(self, a):
        if isinstance(a, int) and 0xD000 <= a < 0xE000:
            return self.mem[1, a]
        return self.mem[a]


class ZeldaEnv:
    def __init__(self, rom, window="null", speed=0):
        # Sans son émulé, le jeu boucle à chaque init du canal 3 en attendant la puce son
        # (bank $39, $49cd et $49eb). On neutralise ces deux sauts : 40 % plus rapide, partie identique.
        sound = window != "null"
        self.pb = PyBoy(rom, window=window, sound_emulated=sound)
        if not sound:
            for a in SOUND_WAIT_JUMPS:
                self.pb.memory[0x39, a] = 0x00
        self.pb.set_emulation_speed(speed)
        self.mem = Ram(self.pb.memory)
        self.held = set()
        self.window = sound
        self.a_item = ITEMS[os.environ["ZELDA_A"]] if os.environ.get("ZELDA_A") else None

    def load(self, state):
        # On relâche tout, pas seulement self.held : l'état des touches après un chargement
        # ne dépend alors plus de la partie d'avant, et un chemin d'inputs se rejoue à l'identique.
        self.pb.load_state(io.BytesIO(state))
        for b in BUTTONS:
            self.pb.button_release(b)
        self.held = set()
        if self.a_item is not None:
            self.equip_a(self.a_item)

    def equip_a(self, item):
        """Met item sur A en l'échangeant avec ce qui y était (dans l'inventaire ou sur B)."""
        m = self.pb.memory
        old = m[EQUIP_A]
        if old == item:
            return
        for a in (EQUIP_B, *STORAGE):
            if m[a] == item:
                m[a] = old
                break
        m[EQUIP_A] = item
        m[0xCBEA] = 0x01  # wStatusBarNeedsRefresh : force la mise à jour visuelle du HUD

    def save(self):
        f = io.BytesIO()
        self.pb.save_state(f)
        return f.getvalue()

    def step(self, wanted, frames, render=False):
        for b in self.held - wanted:
            self.pb.button_release(b)
        for b in wanted - self.held:
            self.pb.button_press(b)
        self.held = set(wanted)
        if self.window:
            # Avec une fenêtre, tick(n) n'affiche que la dernière frame et ne limite la vitesse
            # qu'une fois : écran noir et jeu n fois trop rapide. On avance frame par frame.
            for _ in range(frames):
                alive = self.pb.tick(1, True)
            return alive
        return self.pb.tick(frames, render)

    def frame(self):
        """Écran rendu au dernier tick, moyenne 2x2 en niveaux de gris : 72x80 uint8."""
        s = self.pb.screen.ndarray
        rgb = (s[0::2, 0::2, :3].astype(np.uint16) + s[0::2, 1::2, :3] +
               s[1::2, 0::2, :3] + s[1::2, 1::2, :3]) >> 2
        return ((rgb @ LUMA) >> 8).astype(np.uint8)

    def observe(self):
        m = self.mem
        ty, tx = m[ram.LINK + ram.OBJ_YH] >> 4, m[ram.LINK + ram.OBJ_XH] >> 4
        coll = m[ram.ROOM_COLLISIONS:ram.ROOM_COLLISIONS + ram.ROOM_STRIDE * ram.ROOM_ROWS]
        rows, cols = (11, 15) if m[ram.ROOM_IS_LARGE] else (8, 10)

        # 1 = mur ou hors salle, 0.5 = trou/eau/lave, 0 = libre, -0.5 = PNJ, -1 = ennemi ou projectile.
        grid = np.ones((VIEW, VIEW), dtype=np.float32)
        for gy in range(VIEW):
            y = ty + gy - VIEW_RADIUS
            if not 0 <= y < rows:
                continue
            for gx in range(VIEW):
                x = tx + gx - VIEW_RADIUS
                if 0 <= x < cols:
                    grid[gy, gx] = tile_value(coll[y * ram.ROOM_STRIDE + x])

        # Les ennemis passent après les PNJ pour écraser leur case.
        for base, value in ((ram.INTERACTION_BASE, -0.5), (ram.ENEMY_BASE, -1.0), (ram.PART_BASE, -1.0)):
            for s in range(ram.NUM_SLOTS):
                o = base + s * ram.OBJ_STRIDE
                if not m[o + ram.OBJ_ENABLED]:
                    continue
                if base == ram.INTERACTION_BASE and not m[o + ram.OBJ_VISIBLE] & 0x80:
                    continue
                gy = (m[o + ram.OBJ_YH] >> 4) - ty + VIEW_RADIUS
                gx = (m[o + ram.OBJ_XH] >> 4) - tx + VIEW_RADIUS
                if 0 <= gy < VIEW and 0 <= gx < VIEW:
                    grid[gy, gx] = value

        extra = np.zeros(10, dtype=np.float32)
        extra[0] = m[ram.LINK_HEALTH] / max(1, m[ram.LINK_MAX_HEALTH])
        extra[1] = 1.0 if m[ram.TEXT_ACTIVE] else 0.0
        extra[2 + (m[ram.LINK + ram.OBJ_DIRECTION] & 3)] = 1.0
        extra[6 + (m[ram.SEASON] & 3)] = 1.0
        return np.concatenate((grid.ravel(), extra))
