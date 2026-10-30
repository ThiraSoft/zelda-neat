"""Côté émulateur du PPO, sans torch : épisodes (observation = écran + vecteur, voir obs.py), récompense, départs (Horon, démos reculées,
cellules de l'archive) et processus workers. Chaque worker fait tourner plusieurs émulateurs."""
import json
import os
import random
import signal
from collections import deque

import numpy as np

from . import ram
from .archive import read_state
from .env import ZeldaEnv, EQUIP_B, EQUIP_A, tile_value
from .fitness import Tracker
from .obs import ACTIONS, Observer, observe
from .paths import build_demo
from .runner import FRAMES_PER_ACTION, busy

W_ROOM = 2.0      # salle (groupe, salle) jamais vue pendant l'épisode (hausse pour motiver l'exploration)
W_VARIANT = 0.1   # même salle dans une autre saison : peu, sinon changer de saison sur place rapporte
W_KIND = 1.0      # type d'ennemi tué pour la première fois de l'épisode
W_KILL = 0.1      # chaque ennemi tué, pour qu'il apprenne à se battre
KILL_CAP = 5      # kills payés au plus par salle et par épisode, contre le farm des réapparitions
W_STORY = 1.0     # bit d'histoire gagné
W_TILE = 0.01     # case visitée (réduit de 0.05 à 0.01 pour éviter le piétinement en boucle)
W_BACKTRACK = -0.5  # pénalité si Link refait un demi-tour immédiat vers la salle précédente (< 5s)
W_LOCAL_STAGNATION = -0.02  # pénalité si Link tourne en rond sur place (< 32 px sur 15 pas = 3s)
W_ROOM_STAGNATION = -0.01   # pénalité si Link traîne > 35 pas (7s) dans la même salle sans progrès
W_SPAM = -0.005     # micro-pénalité si appui répété A/B pendant le cooldown (anti-spam bouclier/épée)
COOLDOWN_STEPS = 2  # pas d'action (12 frames = 0.2s par pas) de pause obligatoire entre deux appuis A ou B (max ~1.7/s)
W_EQUIP = 0.02    # micro-bonus la première fois qu'un objet est équipé (exploration menu)
W_RUPEE = 0.005   # micro-bonus par rubis gagné (+0.005 vert, +0.025 bleu, +0.10 rouge)
W_MENU = -0.001     # micro-pénalité par pas dans le menu pour éviter qu'il y reste bloqué
W_DAMAGE = -0.08  # par quart de coeur perdu (incite à esquiver/parer, -0.32 par coeur entier)
W_DEATH = -2.0    # mort : annule entièrement une découverte de salle pour éviter le jeu kamikaze


def bcd_rupees(m):
    return ((m[ram.RUPEES + 1] >> 4) * 1000 + (m[ram.RUPEES + 1] & 0xF) * 100
            + (m[ram.RUPEES] >> 4) * 10 + (m[ram.RUPEES] & 0xF))


class Episode:
    """Un émulateur et l'épisode en cours. Le départ est tiré dans le message de curriculum :
    {"horon": part des départs à Horon, "demos": [(nom, pointeur)], "cells": [states], "jitter": n}."""

    def __init__(self, rom, seed_path, demos_dir, max_steps, patience, rng):
        self.env = ZeldaEnv(rom)
        self.seed = read_state(seed_path)
        self.demos_dir = demos_dir
        self.metas = {}
        self.max_steps = max_steps
        self.patience = patience
        self.rng = rng

    def meta(self, name):
        if name not in self.metas:
            with open(os.path.join(self.demos_dir, name, "meta.json")) as f:
                self.metas[name] = json.load(f)
        return self.metas[name]

    def reset(self, cur):
        demos, cells = cur["demos"], cur.get("cells") or []
        horon = cur["horon"]
        warp = cur.get("warp", 0.0)
        self.demo, self.ptr, self.idx, self.target = None, None, None, None
        prefill_rooms, prefill_kinds, horizon = set(), set(), self.max_steps
        r = self.rng.random()

        if r < horon:
            self.env.load(self.seed)
            step_start = 0
        elif r < horon + warp:
            # Téléportation vers une carte aléatoire du monde entier (Overworld, Subrosia, Donjons)
            self.env.load(self.seed)
            g = self.rng.choice([0, 0, 1, 4, 5])
            room_id = self.rng.randint(0, 63 if g == 1 else 255)
            self.env.pb.memory[0xCC63] = 0x80 | g
            self.env.pb.memory[0xCC64] = room_id
            self.env.pb.memory[0xCC66] = 0x45
            self.env.pb.memory[0xCC67] = 0x01
            self.env.pb.memory[ram.SEASON] = self.rng.randint(0, 3)
            for _ in range(35):
                self.env.step(set(), 1, False)

            m = self.env.mem
            coll = m[ram.ROOM_COLLISIONS:ram.ROOM_COLLISIONS + ram.ROOM_STRIDE * ram.ROOM_ROWS]
            rows, cols = (11, 15) if m[ram.ROOM_IS_LARGE] else (8, 10)
            free = [(y, x) for y in range(1, rows - 1) for x in range(1, cols - 1)
                    if tile_value(coll[y * ram.ROOM_STRIDE + x]) == 0.0]
            if free:
                ty, tx = self.rng.choice(free)
                self.env.pb.memory[ram.LINK + ram.OBJ_YH] = (ty << 4) + 8
                self.env.pb.memory[ram.LINK + ram.OBJ_XH] = (tx << 4) + 8
            step_start = 0
        elif demos:
            self.demo, self.ptr = self.rng.choice(demos)
            self.idx = max(0, self.ptr - self.rng.randint(0, cur["jitter"]))
            m = self.meta(self.demo)
            pts, here, end = m["points"], m["points"][self.idx], m["points"][-1]
            state = read_state(os.path.join(self.demos_dir, self.demo, f"{self.idx}.state.z"))
            prefill_rooms = {tuple(int(x) for x in k.split("-")) for k, i in m["first_seen"].items()
                             if i <= self.idx}
            prefill_kinds = set(here["kinds"])
            # Ce que la démo gagne entre ce point et la fin : à égaler pour un succès.
            self.target = ((end["rooms"] - here["rooms"])
                           + len(set(end["kinds"]) - prefill_kinds) + (end["story"] - here["story"]))
            horizon = min(self.max_steps, 2 * (pts[-1]["step"] - here["step"]) + 450)
            step_start = here["step"]
            self.env.load(state)
        elif cells:
            state = read_state(self.rng.choice(cells))
            step_start = 0
            self.env.load(state)
        else:
            self.env.load(self.seed)
            step_start = 0
        # Limite les coeurs à 4 (16 quarts) pour que la survie et la mort soient un réel enjeu
        self.env.pb.memory[ram.LINK_MAX_HEALTH] = min(self.env.mem[ram.LINK_MAX_HEALTH], 16)
        self.env.pb.memory[ram.LINK_HEALTH] = min(self.env.mem[ram.LINK_HEALTH], 16)
        # Commence à 0 rubis
        self.env.pb.memory[ram.RUPEES] = 0
        self.env.pb.memory[ram.RUPEES + 1] = 0
        # Force le rafraîchissement des icônes d'objets et des rubis dans le HUD
        self.env.pb.memory[0xCBEA] = 0x09
        self.env.step(set(), 1, True)   # une frame rendue pour la première observation
        self.t = Tracker(self.env.mem)
        self.prefill_rooms, self.prefill_kinds = prefill_rooms, prefill_kinds
        self.rooms = set(prefill_rooms)
        self.observer = Observer(prefill_rooms, step=step_start)
        self.kinds = set(prefill_kinds)
        self.room_kills = {}
        self.health = self.env.mem[ram.LINK_HEALTH]
        self.rupees = 0
        self.equipped_items = {self.env.mem[EQUIP_B], self.env.mem[EQUIP_A]} - {0}
        self.current_room = (self.env.mem[ram.ACTIVE_GROUP], self.env.mem[ram.ACTIVE_ROOM])
        self.prev_room = None
        self.room_transition_step = 0
        self.room_steps = 0
        self.pos_history = deque(maxlen=15)
        self.held_a = False
        self.held_b = False
        self.cooldown_a = 0
        self.cooldown_b = 0
        self.horizon = horizon
        self.steps = self.idle = 0
        self.ret = 0.0
        return observe(self.env, self.observer)

    def gain(self):
        return (len(self.rooms - self.prefill_rooms) + len(self.kinds - self.prefill_kinds)
                + self.t.story)

    def step(self, a):
        t = self.t
        n_variants, n_tiles, story, kills = len(t.rooms), len(t.tiles), t.story, t.kills

        # Maintien continu autorisé (bouclier levé ou épée chargée), pause / pénalité après relâchement
        wanted = set(ACTIONS[a])
        spam_penalty = 0.0

        if "a" in wanted:
            if self.held_a:
                # Maintien en cours : pleinement autorisé
                pass
            elif self.cooldown_a > 0:
                # Tentative de ré-appui trop rapide après relâchement
                wanted.remove("a")
                spam_penalty += W_SPAM
                self.cooldown_a -= 1
            else:
                self.held_a = True
        else:
            if self.held_a:
                # Relâchement : déclenche le cooldown post-relâchement
                self.held_a = False
                self.cooldown_a = COOLDOWN_STEPS
            elif self.cooldown_a > 0:
                self.cooldown_a -= 1

        if "b" in wanted:
            if self.held_b:
                # Maintien en cours (charge attaque tourbillon)
                pass
            elif self.cooldown_b > 0:
                # Ré-appui trop rapide
                wanted.remove("b")
                spam_penalty += W_SPAM
                self.cooldown_b -= 1
            else:
                self.held_b = True
        else:
            if self.held_b:
                # Relâchement
                self.held_b = False
                self.cooldown_b = COOLDOWN_STEPS
            elif self.cooldown_b > 0:
                self.cooldown_b -= 1

        self.env.step(wanted, FRAMES_PER_ACTION, True)
        m = self.env.mem
        new = t.update(m, FRAMES_PER_ACTION)
        room = (m[ram.ACTIVE_GROUP], m[ram.ACTIVE_ROOM])
        r = spam_penalty
        if room != self.current_room:
            if room == self.prev_room and (self.steps - self.room_transition_step) < 25:
                # Demi-tour immédiat / aller-retour à la bordure d'écran (< 5 secondes de jeu)
                r += W_BACKTRACK
            self.prev_room = self.current_room
            self.current_room = room
            self.room_transition_step = self.steps
            self.room_steps = 0
        else:
            self.room_steps += 1
        if room not in self.rooms:
            self.rooms.add(room)
            r += W_ROOM
        new_kinds = t.kinds - self.kinds
        r += W_KIND * len(new_kinds)
        self.kinds |= new_kinds
        paid = min(t.kills - kills, KILL_CAP - self.room_kills.get(room, 0))
        if paid > 0:
            self.room_kills[room] = self.room_kills.get(room, 0) + paid
            r += W_KILL * paid
        r += W_VARIANT * (len(t.rooms) - n_variants) + W_TILE * (len(t.tiles) - n_tiles)
        r += W_STORY * (t.story - story)

        # Pénalité de stagnation locale (tourner en rond sur place < 32 px sur 15 pas = 3s)
        pos = (m[ram.LINK + ram.OBJ_YH], m[ram.LINK + ram.OBJ_XH])
        self.pos_history.append(pos)
        if len(self.pos_history) == 15 and not busy(m):
            span = max(max(p[0] for p in self.pos_history) - min(p[0] for p in self.pos_history),
                       max(p[1] for p in self.pos_history) - min(p[1] for p in self.pos_history))
            if span < 32:
                r += W_LOCAL_STAGNATION

        # Pénalité de stagnation en salle (> 35 pas = 7s dans la même pièce sans progrès)
        if self.room_steps > 35 and not busy(m) and not (new or paid > 0):
            r += W_ROOM_STAGNATION

        # Dégâts subis (par quart de coeur)
        hp = m[ram.LINK_HEALTH]
        if hp < self.health:
            r += W_DAMAGE * (self.health - hp)
        self.health = hp

        # Exploration de l'inventaire : récompense la première fois qu'un objet est équipé
        cur_items = {m[EQUIP_B], m[EQUIP_A]} - {0}
        new_items = cur_items - self.equipped_items
        if new_items:
            self.equipped_items |= new_items
            r += W_EQUIP * len(new_items)

        # Micro-bonus pour chaque rubis gagné (ramassé)
        cur_rupees = bcd_rupees(m)
        rupees_gained = cur_rupees - self.rupees
        if rupees_gained > 0:
            r += W_RUPEE * rupees_gained
        self.rupees = cur_rupees

        # Micro-pénalité continue si un menu est ouvert (l'incite à refermer vite)
        if m[ram.OPENED_MENU]:
            r += W_MENU

        if t.dead:
            r += W_DEATH
        self.steps += 1
        self.ret += r
        if new or paid > 0 or new_items or rupees_gained > 0:
            self.idle = 0
        elif not busy(m):
            self.idle += 1
        done = t.dead or self.steps >= self.horizon or self.idle >= self.patience
        info = None
        if done:
            g = self.gain()
            info = {"demo": self.demo, "ptr": self.ptr, "idx": self.idx, "gain": g,
                    "target": self.target, "success": self.target is not None and g >= self.target,
                    "score": len(self.rooms - self.prefill_rooms) + len(self.kinds - self.prefill_kinds),
                    "story": t.story, "kills": t.kills, "steps": self.steps, "return": self.ret, "dead": t.dead}
        return observe(self.env, self.observer), r, done, info


def worker(conn, rom, seed_path, demos_dir, n_envs, max_steps, patience, seed):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    rng = random.Random(seed)
    envs = [Episode(rom, seed_path, demos_dir, max_steps, patience, rng) for _ in range(n_envs)]
    cur = None
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            cur = data
            obs = [e.reset(cur) for e in envs]
            conn.send((np.stack([o[0] for o in obs]), np.stack([o[1] for o in obs])))
        elif cmd == "step":
            # Le curriculum n'est envoyé que quand il change.
            actions, new_cur = data
            cur = new_cur or cur
            obs, rew, done, infos = [], [], [], []
            for e, a in zip(envs, actions):
                o, r, d, info = e.step(int(a))
                if d:
                    infos.append(info)
                    o = e.reset(cur)
                obs.append(o)
                rew.append(r)
                done.append(d)
            conn.send((np.stack([o[0] for o in obs]), np.stack([o[1] for o in obs]),
                       np.array(rew, np.float32), np.array(done, np.float32), infos))
        else:
            break


class VecEnv:
    def __init__(self, ctx, n_workers, envs_per_worker, rom, seed_path, demos_dir, max_steps, patience):
        self.conns, self.procs = [], []
        for i in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, daemon=True,
                            args=(b, rom, seed_path, demos_dir, envs_per_worker, max_steps, patience,
                                  random.getrandbits(32)))
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def reset(self, cur):
        for c in self.conns:
            c.send(("reset", cur))
        out = [c.recv() for c in self.conns]
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    def step(self, actions, cur=None):
        chunks = np.array_split(actions, len(self.conns))
        for c, a in zip(self.conns, chunks):
            c.send(("step", (a, cur)))
        out = [c.recv() for c in self.conns]
        infos = [i for o in out for i in o[4]]
        return tuple(np.concatenate([o[k] for o in out]) for k in range(4)) + (infos,)

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except OSError:
                pass
        for p in self.procs:
            p.join(timeout=5)


class Curriculum:
    """Phase 2 de Go-Explore façon "backward algorithm" : chaque démo commence près de sa fin.
    Quand assez d'épisodes partis du pointeur font au moins aussi bien que la démo, le pointeur
    recule vers Horon. Une démo résolue depuis son début est remplacée par une nouvelle cible."""

    def __init__(self, horon, back, success, window, jitter, warp=0.0):
        self.horon, self.back, self.success, self.window, self.jitter = horon, back, success, window, jitter
        self.warp = warp
        self.demos = {}   # nom -> {"ptr", "points", "results", "solved"}
        self.cells = []

    def add(self, name, n_points, ptr=None):
        self.demos[name] = {"ptr": max(0, n_points - 1 - self.back) if ptr is None else ptr,
                            "points": n_points, "results": deque(maxlen=self.window), "solved": False}

    def message(self):
        return {"horon": self.horon, "warp": self.warp, "jitter": self.jitter, "cells": self.cells,
                "demos": [(n, d["ptr"]) for n, d in self.demos.items() if not d["solved"]]}

    def record(self, info):
        d = self.demos.get(info["demo"])
        if d is None or d["solved"] or info["ptr"] != d["ptr"]:
            return None
        d["results"].append(info["success"])
        if len(d["results"]) < self.window or sum(d["results"]) < self.success * self.window:
            return None
        d["results"].clear()
        if d["ptr"] == 0:
            d["solved"] = True
            return "résolue"
        d["ptr"] = max(0, d["ptr"] - self.back)
        return "recule"

    def state(self):
        return {"horon": self.horon, "demos": {n: {"ptr": d["ptr"], "points": d["points"],
                                                    "solved": d["solved"]} for n, d in self.demos.items()}}

    def summary(self):
        active = [f"{n} {d['ptr']}/{d['points'] - 1} ({100 * sum(d['results']) // max(1, len(d['results']))} %)"
                  for n, d in self.demos.items() if not d["solved"]]
        solved = sum(d["solved"] for d in self.demos.values())
        return ", ".join(active) + (f", {solved} résolues depuis Horon" if solved else "")


def make_demo(args):
    """Pour un Pool : (rom, state de la graine, inputs, dossier, every) -> (dossier, meta)."""
    rom, seed_path, actions, out_dir, every = args
    env = ZeldaEnv(rom)
    return out_dir, build_demo(env, read_state(seed_path), actions, out_dir, every)
