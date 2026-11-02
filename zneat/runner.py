import os
import pickle
import random
import signal
import zlib

import neat

from . import ram
from .archive import cell_key, read_state
from .env import BUTTONS, EQUIP_A, EQUIP_B, ZeldaEnv, to_mask, warp_random
from .fitness import Tracker
from .obs import NeatObserver

FRAMES_PER_ACTION = 12     # 5 décisions par seconde
MAX_FRAMES = 2 * 60 * 60   # 2 minutes de jeu par save state
PATIENCE = 15 * 60         # arrêt si rien de nouveau pendant 15 s (voir strict dans rollout)

# Exploration aléatoire façon Go-Explore : une action tirée au hasard est répétée
# tant qu'un tirage reste sous STICKY (en moyenne 40 frames à 0.7, soit 3,3 pas de 12 frames).
RANDOM_ACTIONS = (set(), {"up"}, {"down"}, {"left"}, {"right"}, {"a"}, {"b"},
                  {"up", "a"}, {"down", "a"}, {"left", "a"}, {"right", "a"})
STICKY = 0.7

_env = None
_states = None
_config = None
_cache = {}    # chemin -> state lu, pour les départs d'une génération (vidé quand il grossit)
_known = (None, frozenset())   # (mtime de known.pkl, clés) : rechargé quand l'archive change


def make_net(genome, config):
    if config.genome_config.feed_forward:
        return neat.nn.FeedForwardNetwork.create(genome, config)
    return neat.nn.RecurrentNetwork.create(genome, config)


def busy(m):
    """Dialogue ou cinématique : Link ne peut pas avancer, la patience est en pause."""
    return m[ram.TEXT_ACTIVE] or m[ram.DISABLED_OBJECTS] & 0x81


def rollout(env, policy, state, render=False, max_frames=None, patience=None, known=None, on_step=None, warp=False, rng=None,
            strict=False):
    """Joue une partie avec policy(obs) -> boutons. Avec `known` (clés de l'archive),
    t.captured reçoit par cellule nouvelle (state compressé, inputs depuis le départ), pris
    quand Link est contrôlable. Un input = un octet (voir env.to_mask) tenu FRAMES_PER_ACTION frames.
    strict : la patience ne repart que sur une salle, un type tué dans la salle ou de l'histoire
    (comme le PPO), pas sur une case nouvelle. Une politique qui a new_game() est remise à zéro."""
    if hasattr(policy, "new_game"):
        policy.new_game()
    max_frames = max_frames or MAX_FRAMES
    patience = patience or PATIENCE
    env.load(state)
    if warp:
        warp_random(env, rng)
    else:
        m = env.pb.memory
        m[ram.LINK_MAX_HEALTH] = min(env.mem[ram.LINK_MAX_HEALTH], 16)
        m[ram.LINK_HEALTH] = min(env.mem[ram.LINK_HEALTH], 16)
        m[ram.RUPEES] = 0
        m[ram.RUPEES + 1] = 0
        m[EQUIP_B] = 0x05
        m[EQUIP_A] = 0x01
        m[0xCBEA] = 0x09
        env.step(set(), 1, render)
    t = Tracker(env.mem, known or frozenset(), cell_key if known is not None else None)
    t.captured = {}
    actions = bytearray()
    idle = 0
    while t.frames < max_frames and idle < patience and not t.dead:
        buttons = policy(env)
        actions.append(to_mask(buttons))
        if not env.step(buttons, FRAMES_PER_ACTION, render):
            break
        m = env.mem
        new = t.update(m, FRAMES_PER_ACTION)
        if t.progress if strict else new:
            idle = 0
        elif not busy(m):
            idle += FRAMES_PER_ACTION
        if (known is not None and t.cell in t.novel and t.cell not in t.captured
                and not t.dead and m[ram.LINK_HEALTH] and not busy(m)):
            t.captured[t.cell] = (zlib.compress(env.save(), 1), bytes(actions))
        if on_step is not None:
            if on_step(env, t, buttons) is False:
                break
    return t


class net_policy:
    """Joueur NEAT : 6 sorties (haut, bas, gauche, droite, A, B), bouton tenu au-dessus de 0,5."""

    def __init__(self, net):
        self.net = net
        self.observer = NeatObserver()

    def new_game(self):
        if hasattr(self.net, "reset"):
            self.net.reset()
        self.observer.new_game()

    def __call__(self, env):
        return {b for b, v in zip(BUTTONS, self.net.activate(self.observer(env))) if v > 0.5}


def random_policy(rng):
    cur = [set()]

    def policy(env):
        if rng.random() > STICKY:
            cur[0] = rng.choice(RANDOM_ACTIONS)
        return cur[0]
    return policy


def play(env, net, state, **kw):
    return rollout(env, net_policy(net), state, **kw)


def solo(env, policy, state, seconds=1200, render=False, on_step=None, warp=False, rng=None):
    """Métrique joueur seul : une partie de `seconds` depuis `state`, sans save state ni
    patience, arrêtée seulement par la mort. Renvoie le Tracker (voir solo_line)."""
    frames = int(seconds * 60)
    return rollout(env, policy, state, render=render, max_frames=frames, patience=frames + 1, on_step=on_step, warp=warp, rng=rng)


def solo_score(t):
    """Salles (groupe, salle), sans compter la saison, plus types d'ennemis tués."""
    return len({r[:2] for r in t.rooms}) + len(t.kinds)


def solo_line(t):
    rooms = len({r[:2] for r in t.rooms})
    return (f"score {solo_score(t)} = {rooms} salles + {len(t.kinds)} types d'ennemis, "
            f"{len(t.rooms)} salles x saison, histoire +{t.story}, "
            f"{t.frames / 60:.0f} s{' (mort)' if t.dead else ''}")


def init_worker(rom, state_paths=(), max_seconds=None, patience_seconds=None, config_path=None):
    global _env, _states, _config, MAX_FRAMES, PATIENCE
    if config_path:
        _config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    if max_seconds:
        MAX_FRAMES = int(max_seconds * 60)
    if patience_seconds:
        PATIENCE = int(patience_seconds * 60)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _env = ZeldaEnv(rom)
    _states = [read_state(p) for p in state_paths]


def known_cells(path):
    global _known
    mtime = os.stat(path).st_mtime_ns
    if _known[0] != mtime:
        with open(path, "rb") as f:
            _known = (mtime, pickle.load(f))
    return _known[1]


def eval_genome(genome, config):
    net = make_net(genome, config)
    return sum(play(_env, net, s).fitness() for s in _states) / len(_states)


def job(args):
    """Un travail du pool. ("neat", genome, config, départs) -> (fitness, cellules capturées).
    ("random", départ, graine, frames) -> (None, cellules capturées).
    Un départ est (clé, chemin du state) ; une cellule capturée est clé -> (state, clé parente, inputs).
    Les deux renvoient aussi (kills, types d'ennemis tués).
    ("solo", genome, config, state, secondes) -> Tracker de la partie solo, voir solo()."""
    kind, known_path = args[0], args[1]
    if kind == "game":
        # Un génome, un départ téléporté (state sur disque) : (fitness, kills, types tués), patience stricte.
        genome, path = args[2:]
        if path not in _cache:
            if len(_cache) > 64:
                _cache.clear()
            _cache[path] = read_state(path)
        t = play(_env, make_net(genome, _config), _cache[path], strict=True)
        return t.fitness(), t.kills, t.kinds
    if kind == "eval":
        # Une partie de l'évaluation commune (zneat/evaluate.py) depuis _states[0].
        from .evaluate import eval_game
        genome, g, seconds = args[2:]
        return eval_game(_env, net_policy(make_net(genome, _config)), _states[0], g, seconds)
    if kind == "solo":
        genome, config, path, seconds = args[2:]
        t = solo(_env, net_policy(make_net(genome, config)), read_state(path), seconds)
        t.captured = None
        return t
    # Sans archive (départs au hasard, score solo) : rien n'est capturé, pas de bonus de nouveauté.
    known = known_cells(known_path) if known_path else None
    captured = {}
    if kind == "neat":
        genome, config, starts = args[2:]
        net = make_net(genome, config)
        fit, kills, kinds = 0.0, 0, set()
        for key, p in starts:
            t = play(_env, net, read_state(p), known=known)
            fit += t.fitness()
            kills += t.kills
            kinds |= t.kinds
            for k, (z, acts) in t.captured.items():
                captured.setdefault(k, (z, key, acts))
        return fit / len(starts), captured, kills, kinds
    (key, start), seed, frames = args[2:]
    t = rollout(_env, random_policy(random.Random(seed)), read_state(start),
                max_frames=frames, known=known)
    return None, {k: (z, key, acts) for k, (z, acts) in t.captured.items()}, t.kills, t.kinds


def indexed_job(item):
    i, args = item
    return i, job(args)
