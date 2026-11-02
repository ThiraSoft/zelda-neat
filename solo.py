"""Partie solo : un génome NEAT ou un modèle PPO part de Horon et joue seul, sans save state.
Score = salles (groupe, salle) visitées + types d'ennemis tués.

    solo.py --genome runs/horon/best.pkl
    solo.py --model runs/ppo/model.pt [--greedy] [--window] [--term]
    solo.py --model models/ppo.pt --random-location [spawns]   # départ dans une cellule au hasard
"""
import argparse
import random
import warnings

warnings.filterwarnings("ignore")

from zneat.archive import read_state
from zneat.env import ZeldaEnv
from zneat.runner import FRAMES_PER_ACTION, solo, solo_line

ap = argparse.ArgumentParser()
ap.add_argument("--rom", default="rom/oos.gbc")
ap.add_argument("--state", default="states/horon-endgame.state")
ap.add_argument("--config", default="config-neat.ini")
ap.add_argument("--genome", help="génome NEAT (.pkl)")
ap.add_argument("--model", help="modèle PPO (.pt)")
ap.add_argument("--greedy", action="store_true", help="PPO : action la plus probable au lieu d'échantillonner")
ap.add_argument("--seed", type=int, default=0, help="PPO : graine de l'échantillonnage")
ap.add_argument("--seconds", type=float, default=300)
ap.add_argument("--window", action="store_true", help="affiche le jeu (vitesse réelle)")
ap.add_argument("--term", action="store_true", help="rendu direct dans le terminal (ANSI truecolor, compatible neonmux/PWA)")
ap.add_argument("--cols", type=int, default=0, help="largeur en colonnes pour --term (40=compact, 80=exactement ce que voit le modèle, 0=auto)")
ap.add_argument("--fps", type=float, default=60 / FRAMES_PER_ACTION, help="vitesse d'affichage pour --term (défaut : vitesse réelle, 0=illimité)")
ap.add_argument("--loop", action="store_true", help="rejoue en boucle continue en rechargeant le modèle le plus récent à chaque partie")
ap.add_argument("--random-location", nargs="?", const="spawns", metavar="DOSSIER",
                help="part d'une cellule tirée au hasard dans DOSSIER (défaut spawns), comme dataset.py")
ap.add_argument("--warp", action="store_true", help="parachutage sur une carte aléatoire du monde entier (comme à l'entraînement 100%% warp)")
args = ap.parse_args()
if args.random_location:
    import glob
    import os
    from dataset import room_capped
    folder = args.random_location
    if not os.path.exists(folder) or not glob.glob(os.path.join(folder, "*.state*")):
        folder = "runs/horon/archive"
    candidates = room_capped(folder, 8)
    if candidates:
        args.state = random.Random(args.seed).choice(candidates)
        print("départ :", args.state, flush=True)

# La fenêtre SDL s'ouvre avant de charger torch : dans l'autre ordre, elle reste noire (ROCm).
env = ZeldaEnv(args.rom, window="SDL2" if args.window else "null", speed=1 if args.window else 0)

model_tag = "Zelda PPO"
def get_model_and_tag(model_path, greedy, seed):
    tag = "Zelda PPO"
    ckpt_file = os.path.join(os.path.dirname(model_path), "ckpt.pt")
    if os.path.exists(ckpt_file):
        try:
            import torch
            ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            st = ck.get("step")
            if st:
                tag = f"PPO {st / 1e6:.1f}M"
        except Exception:
            pass
    from zneat.ppo_model import load_model, torch_policy
    pol = torch_policy(load_model(model_path), greedy, seed)
    return pol, tag

if args.model:
    import os
    policy, model_tag = get_model_and_tag(args.model, args.greedy, args.seed)
elif args.genome:
    import pickle

    import neat

    from zneat.runner import make_net, net_policy
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)
    policy = net_policy(make_net(pickle.load(open(args.genome, "rb")), config))
    model_tag = "NEAT"
else:
    raise SystemExit("--genome ou --model")

# Le modèle PPO regarde l'écran : il faut le rendre à chaque pas.
viewer = None
if args.term:
    from zneat.term import TermViewer
    viewer = TermViewer(cols=args.cols, fps=args.fps, tag=model_tag)

last_mtime = [os.path.getmtime(args.model) if args.model and os.path.exists(args.model) else 0]

def step_callback(env, t, buttons):
    if viewer:
        viewer.render(env, t, buttons)
    # Vérification toutes les 0.5s (30 frames) : si un nouveau checkpoint arrive, on relance immédiatement
    if args.loop and args.model and t.frames % 30 == 0:
        try:
            mt = os.path.getmtime(args.model)
            if mt > last_mtime[0]:
                last_mtime[0] = mt
                return False  # Interrompt l'épisode pour recharger le nouveau modèle
        except OSError:
            pass
    return True

episode = 0
try:
    while True:
        episode += 1
        if episode > 1 and args.model:
            policy, model_tag = get_model_and_tag(args.model, args.greedy, args.seed + episode)
            if os.path.exists(args.model):
                last_mtime[0] = os.path.getmtime(args.model)
            if viewer:
                viewer.tag = model_tag

        state_path = args.state
        if args.random_location and candidates:
            state_path = random.choice(candidates)

        t = solo(
            env,
            policy,
            read_state(state_path),
            args.seconds,
            render=args.window or bool(args.model) or args.term,
            on_step=step_callback,
            warp=args.warp,
        )

        if not args.loop:
            break
finally:
    if viewer:
        viewer.cleanup()

if not viewer or not args.loop:
    print("solo :", solo_line(t))
