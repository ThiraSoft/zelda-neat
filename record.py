"""Enregistrement d'une démonstration humaine pour l'apprentissage par imitation (Behavioral Cloning).

Ouvre une fenêtre PyBoy à 60 fps en vitesse réelle avec contrôles clavier ou manette.
Enregistre à chaque pas de 12 frames (0,20 s) :
  - frames.npy (N, 72, 80) uint8 : écran en niveaux de gris sous-échantillonné 2x2
  - vec.npy    (N, 123) float32  : vecteur d'observation PPO (Observer)
  - act.npy    (N,) int64        : indice de l'action jouée dans obs.ACTIONS
  - meta.json  statistiques de la partie (salles visitées, durée, monstres tués)
  - start.state.z et end.state.z

Contrôles clavier par défaut dans la fenêtre :
  - Flèches : Déplacements (Haut, Bas, Gauche, Droite)
  - Z ou A   : Bouton A (Bouclier)
  - X ou S   : Bouton B (Épée)
  - Entrée   : Start (Menu / Pause)
  - Retour   : Select (Carte)
  - Échap    : Sauvegarder et quitter

Usage :
    python record.py
    python record.py --name session-01
    python record.py --scale 4 --start states/horon-endgame.state
    make record
"""
import argparse
from datetime import datetime
import json
import os
import signal
import sys
import time
import zlib

import numpy as np

from zneat.archive import read_state
from zneat.env import EQUIP_A, EQUIP_B, ZeldaEnv
from zneat.fitness import Tracker
from zneat.obs import ACTIONS, Observer, observe
from zneat.paths import FRAMES_PER_ACTION
import zneat.ram as ram

# Table de correspondance inversée : ensemble de boutons -> index d'action PPO
ACTION_MAP = {frozenset(a): i for i, a in enumerate(ACTIONS)}


def gb_keys(byte_val):
    """Convertit l'octet de touches de la Game Boy (WRAM $C480) en ensemble de boutons."""
    buttons = set()
    if byte_val & 0x01:
        buttons.add("a")
    if byte_val & 0x02:
        buttons.add("b")
    if byte_val & 0x08:
        buttons.add("start")
    if byte_val & 0x10:
        buttons.add("right")
    if byte_val & 0x20:
        buttons.add("left")
    if byte_val & 0x40:
        buttons.add("up")
    if byte_val & 0x80:
        buttons.add("down")
    return buttons


def map_buttons_to_action(b_set):
    """Mappe n'importe quelle combinaison de touches humaines vers les 16 actions discrètes."""
    fs = frozenset(b_set)
    if fs in ACTION_MAP:
        return ACTION_MAP[fs]
    # En cas de diagonale ou de combinaison non standard, priorité direction + item
    dirs = [d for d in ("up", "down", "left", "right") if d in b_set]
    has_a = "a" in b_set
    has_b = "b" in b_set
    if dirs:
        d = dirs[0]
        if has_b and frozenset({d, "b"}) in ACTION_MAP:
            return ACTION_MAP[frozenset({d, "b"})]
        if has_a and frozenset({d, "a"}) in ACTION_MAP:
            return ACTION_MAP[frozenset({d, "a"})]
        return ACTION_MAP[frozenset({d})]
    if has_b:
        return ACTION_MAP[frozenset({"b"})]
    if has_a:
        return ACTION_MAP[frozenset({"a"})]
    if "start" in b_set:
        return ACTION_MAP[frozenset({"start"})]
    return 0  # set() neutre


def main():
    ap = argparse.ArgumentParser(description="Enregistrement d'une démonstration humaine pour PPO / Imitation")
    ap.add_argument("--rom", default="rom/oos.gbc", help="ROM de Zelda Oracle of Seasons")
    ap.add_argument("--start", default="states/horon-endgame.state", help="Save state de départ")
    ap.add_argument("--out", default="demos/human", help="Dossier parent où enregistrer la démonstration")
    ap.add_argument("--name", default=None, help="Nom du sous-dossier (défaut : date et heure)")
    ap.add_argument("--scale", type=int, default=3, help="Facteur de zoom de la fenêtre (défaut : 3 = 480x432)")
    args = ap.parse_args()

    tag = args.name or datetime.now().strftime("session-%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("  🎮 ENREGISTREMENT D'UNE DÉMONSTRATION HUMAINE - ZELDA OOS")
    print("=" * 65)
    print("  Contrôles :")
    print("    • Flèches : Déplacements")
    print("    • Z ou A  : Bouclier (Bouton A)")
    print("    • X ou S  : Épée (Bouton B)")
    print("    • Entrée  : Start (Menu / Pause)")
    print("    • Échap   : Sauvegarder et quitter")
    print(f"  Dossier cible : {out_dir}")
    print("=" * 65, flush=True)

    # Initialisation de PyBoy avec rendu SDL2 à vitesse 1 (60 fps réel)
    env = ZeldaEnv(args.rom, window="SDL2", speed=1)
    # Définition du zoom
    env.pb.set_emulation_speed(1)

    # Chargement du state de départ
    if not os.path.exists(args.start):
        raise SystemExit(f"Fichier de départ introuvable : {args.start}")
    raw_state = read_state(args.start)
    env.load(raw_state)

    # Calibrage : 4 cœurs max, 0 rubis, Épée B, Bouclier A et rafraîchissement HUD
    m = env.pb.memory
    m[ram.LINK_MAX_HEALTH] = min(env.mem[ram.LINK_MAX_HEALTH], 16)
    m[ram.LINK_HEALTH] = min(env.mem[ram.LINK_HEALTH], 16)
    m[ram.RUPEES] = 0
    m[ram.RUPEES + 1] = 0
    m[EQUIP_B] = 0x05  # Épée
    m[EQUIP_A] = 0x01  # Bouclier
    m[0xCBEA] = 0x09   # Forcer rafraîchissement icônes + rubis dans le HUD

    # Exécution de quelques frames pour que le moteur Game Boy valide le HUD
    for _ in range(15):
        env.step(set(), 1, True)

    # Sauvegarde du state initial calibré
    with open(os.path.join(out_dir, "start.state.z"), "wb") as f:
        f.write(zlib.compress(env.save(), 1))

    observer = Observer()
    tracker = Tracker(env.mem)

    frames_buf = []
    vecs_buf = []
    acts_buf = []

    running = True

    def stop_handler(signum, frame):
        nonlocal running
        print("\nArrêt demandé...")
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    step_count = 0
    t_start = time.time()
    t_last_log = t_start

    try:
        while running:
            # 1. Observation avant action
            f = env.frame()
            v = observer(env)

            # 2. Avance de FRAMES_PER_ACTION (12 frames = 0,20 s) en lisant les entrées utilisateur
            held_union = set()
            alive = True
            for _ in range(FRAMES_PER_ACTION):
                alive = env.pb.tick(1, True)
                if not alive:
                    running = False
                    break
                # Lecture des touches actives dans la WRAM de la Game Boy
                held_union |= gb_keys(env.mem[0xC480])

            if not alive:
                break

            # 3. Détermination de l'action enregistrée
            act_idx = map_buttons_to_action(held_union)

            frames_buf.append(f)
            vecs_buf.append(v)
            acts_buf.append(act_idx)

            tracker.update(env.mem, FRAMES_PER_ACTION)
            step_count += 1

            # Affichage périodique de progression (toutes les 30 s)
            now = time.time()
            if now - t_last_log >= 30:
                elapsed = now - t_start
                mins = elapsed / 60
                rooms_count = len({r[:2] for r in tracker.rooms})
                hp = env.mem[ram.LINK_HEALTH] // 4
                print(f"⏱️  {mins:4.1f} min | {step_count:5d} pas | Salles : {rooms_count:2d} | "
                      f"Kills : {tracker.kills:2d} | Santé : {hp}/4 cœurs | Touches : {held_union}",
                      flush=True)
                t_last_log = now

            # Arrêt si Link meurt
            if env.mem[ram.LINK_HEALTH] == 0:
                print("\n⚠️ Link est mort ! Fin de l'enregistrement.")
                break

    except KeyboardInterrupt:
        pass

    total_time = time.time() - t_start
    if not frames_buf:
        print("Aucun pas enregistré.")
        return

    print("\nFinalisation de la sauvegarde des fichiers...")
    arr_frames = np.array(frames_buf, dtype=np.uint8)
    arr_vecs = np.array(vecs_buf, dtype=np.float32)
    arr_acts = np.array(acts_buf, dtype=np.int64)

    np.save(os.path.join(out_dir, "frames.npy"), arr_frames)
    np.save(os.path.join(out_dir, "vec.npy"), arr_vecs)
    np.save(os.path.join(out_dir, "act.npy"), arr_acts)

    # Sauvegarde du state final
    with open(os.path.join(out_dir, "end.state.z"), "wb") as f:
        f.write(zlib.compress(env.save(), 1))

    meta = {
        "tag": tag,
        "date": datetime.now().isoformat(),
        "steps": step_count,
        "frames_total": step_count * FRAMES_PER_ACTION,
        "duration_seconds": round(total_time, 1),
        "duration_minutes": round(total_time / 60, 2),
        "rooms": len({r[:2] for r in tracker.rooms}),
        "rooms_list": [list(r) for r in sorted({r[:2] for r in tracker.rooms})],
        "kills": tracker.kills,
        "kinds": len(tracker.kinds),
        "story": tracker.story,
        "dead": env.mem[ram.LINK_HEALTH] == 0,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 65)
    print(f"✅ DÉMONSTRATION ENREGISTRÉE AVEC SUCCÈS DANS :")
    print(f"   {out_dir}")
    print(f"   • Pas d'action   : {step_count} (environ {total_time / 60:.1f} minutes)")
    print(f"   • Salles explorées: {len({r[:2] for r in tracker.rooms})}")
    print(f"   • Monstres tués   : {tracker.kills}")
    print(f"   • Fichiers prêts  : frames.npy ({arr_frames.nbytes / 1e6:.1f} Mo), vec.npy, act.npy, meta.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
