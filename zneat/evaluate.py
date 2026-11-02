"""Évaluation commune au PPO et à NEAT : parties téléportées à graines fixes, comparables d'un run à l'autre.
La partie g part de warp_random(Random(g)) et dure `seconds`, arrêtée seulement par la mort."""
import random
from collections import Counter

from . import ram
from .env import EQUIP_A, EQUIP_B
from .runner import solo

EVAL_GAMES = 30
EVAL_SECONDS = 120


def eval_game(env, policy, state, g, seconds=EVAL_SECONDS, render=False):
    """Une partie d'évaluation : dictionnaire de compteurs, à cumuler avec summarize."""
    c = {"steps": 0, "menu": 0, "dialog": 0, "acts": Counter(), "equips": set()}

    def on_step(e, t, buttons):
        c["steps"] += 1
        c["menu"] += bool(e.mem[ram.OPENED_MENU])
        c["dialog"] += bool(e.mem[ram.TEXT_ACTIVE])
        c["acts"]["+".join(sorted(buttons)) or "rien"] += 1
        c["equips"].add((e.mem[EQUIP_B], e.mem[EQUIP_A]))

    t = solo(env, policy, state, seconds, render=render, on_step=on_step, warp=True, rng=random.Random(g))
    c.update(rooms=len({r[:2] for r in t.rooms}), kinds=len(t.kinds), kills=t.kills, dead=t.dead,
             seconds=t.frames / 60, swap=len(c.pop("equips")) > 1)
    return c


def summarize(games):
    n = len(games)
    steps = max(1, sum(c["steps"] for c in games))
    acts = sum((c["acts"] for c in games), Counter())
    kinds = sum(c["kinds"] for c in games)
    kills = sum(c["kills"] for c in games)
    return {"games": n, "salles": sum(c["rooms"] for c in games) / n, "salles_max": max(c["rooms"] for c in games),
            "types": kinds / n, "kills": kills / n, "kills_par_type": kills / max(1, kinds),
            "morts_pct": 100 * sum(c["dead"] for c in games) / n,
            "duree_s": sum(c["seconds"] for c in games) / n,
            "menu_pct": 100 * sum(c["menu"] for c in games) / steps,
            "dialogue_pct": 100 * sum(c["dialog"] for c in games) / steps,
            "items_changes": sum(c["swap"] for c in games),
            "actions_pct": {a: round(100 * k / steps, 1) for a, k in acts.most_common(8)}}


def eval_score(res):
    """Un nombre pour classer : salles plus types d'ennemis tués, en moyenne par partie."""
    return res["salles"] + res["types"]
