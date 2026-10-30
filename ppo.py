"""Un seul réseau (CNN sur l'écran + LSTM) entraîné par PPO, avec départs tirés de l'archive
Go-Explore puis reculés vers Horon, et imitation des chemins de l'archive (pré-entraînement à chaque
nouvelle démo, puis perte d'imitation ajoutée au PPO). Les émulateurs tournent dans des workers CPU,
le réseau sur le GPU s'il y en a un.

    ppo.py --archive runs/horon/archive --run runs/ppo --workers 7
    ppo.py ... --resume       # reprend runs/ppo/ckpt.pt
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import time


def build_demos(ctx, args, archive, targets, curriculum):
    from zneat.archive import key_str
    from zneat.ppo_env import make_demo
    jobs = []
    for k in targets:
        name = key_str(k)
        out = os.path.join(args.run, "demos", name)
        if not os.path.exists(os.path.join(out, "meta.json")):
            seed, actions = archive.full_path(k)
            jobs.append((args.rom, archive.path(seed), actions, out, args.demo_every))
    if jobs:
        with ctx.Pool(min(len(jobs), args.workers)) as pool:
            for out, meta in pool.imap_unordered(make_demo, jobs):
                print(f"démo {os.path.basename(out)} : {meta['points'][-1]['step']} pas, "
                      f"{meta['points'][-1]['rooms']} salles, arrivée {meta['end']}")
    for k in targets:
        name = key_str(k)
        meta = json.load(open(os.path.join(args.run, "demos", name, "meta.json")))
        curriculum.add(name, len(meta["points"]))


def new_targets(archive, curriculum, n):
    from zneat.archive import key_str
    from zneat.paths import pick_targets
    used = set(curriculum.demos)
    return [k for k in pick_targets(archive, n + len(used)) if key_str(k) not in used][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="rom/oos.gbc")
    ap.add_argument("--start", default="states/horon-endgame.state", help="départ de la partie solo (Horon)")
    ap.add_argument("--archive", default="runs/horon/archive")
    ap.add_argument("--run", default="runs/ppo")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--envs-per-worker", type=int, default=4)
    ap.add_argument("--steps", type=int, default=256, help="pas par env entre deux mises à jour")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--arch", choices=("nature", "impala"), default="impala",
                    help="impala : CNN résiduel plus profond, fc 512")
    ap.add_argument("--max-seconds", type=float, default=1200, help="durée max d'un épisode")
    ap.add_argument("--patience", type=float, default=120, help="fin d'épisode après N s sans rien de nouveau")
    ap.add_argument("--horon", type=float, default=0.2, help="part des épisodes qui partent de Horon")
    ap.add_argument("--demos", type=int, default=4, help="cibles de l'archive suivies en même temps")
    ap.add_argument("--demo-every", type=int, default=32, help="un state de démo tous les N pas")
    ap.add_argument("--back-seconds", type=float, default=10, help="recul du départ à chaque palier")
    ap.add_argument("--success", type=float, default=0.25, help="taux de succès pour reculer")
    ap.add_argument("--window", type=int, default=16, help="épisodes jugés avant de reculer")
    ap.add_argument("--refresh-minutes", type=float, default=10,
                    help="relit l'archive toutes les N min pour compléter les démos")
    ap.add_argument("--solo-minutes", type=float, default=15, help="partie solo toutes les N min")
    ap.add_argument("--solo-seconds", type=float, default=1200)
    ap.add_argument("--bc-burst", type=int, default=0, help="pas d'imitation à chaque nouvelle démo")
    ap.add_argument("--bc-coef", type=float, default=0.0, help="poids de l'imitation dans la perte PPO")
    ap.add_argument("--bc-batch", type=int, default=16, help="fenêtres de démo par lot d'imitation")
    ap.add_argument("--bc-len", type=int, default=64, help="longueur d'une fenêtre de démo")
    ap.add_argument("--total-steps", type=float, default=None, help="arrêt après N pas (vide = jamais)")
    ap.add_argument("--cells-per-room", type=int, default=8,
                    help="sans démo : cellules de départ gardées par salle de l'archive")
    ap.add_argument("--init", default=None, help="ckpt.pt dont on reprend les poids pour un nouveau run")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch.distributions import Categorical

    from zneat.archive import Archive
    from zneat.ppo_env import Curriculum, VecEnv
    from zneat.env import SCREEN
    from zneat.obs import PPO_INPUTS
    from zneat.ppo_model import DemoData, Policy, bc_loss, save_model, solo_eval
    from zneat.runner import FRAMES_PER_ACTION

    ctx = mp.get_context("spawn")
    os.makedirs(args.run, exist_ok=True)
    ckpt_path = os.path.join(args.run, "ckpt.pt")
    model_path = os.path.join(args.run, "model.pt")
    if os.path.exists(ckpt_path) and not args.resume:
        raise SystemExit(f"{args.run} contient déjà un entraînement : --resume ou autre --run")

    back = max(1, round(args.back_seconds * 60 / FRAMES_PER_ACTION / args.demo_every))
    curriculum = Curriculum(args.horon, back, args.success, args.window, jitter=min(2, back))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Policy(hidden=args.hidden, arch=args.arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get("kind") != "cnn-lstm" or ck.get("n_vec") != PPO_INPUTS or ck.get("arch", "nature") != args.arch:
            raise SystemExit(f"{ckpt_path} : ancienne version du réseau, make ppo-reset CONFIRM=oui puis relancer")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        for name, d in ck["curriculum"]["demos"].items():
            curriculum.add(name, d["points"], d["ptr"])
            curriculum.demos[name]["solved"] = d["solved"]
        print(f"reprise à l'étape {step}")
    elif args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["model"])
        print(f"poids repris de {args.init}")

    def refresh():
        """Relit l'archive (l'explorateur peut tourner en même temps) et complète les démos."""
        archive = Archive(args.archive)
        active = sum(not d["solved"] for d in curriculum.demos.values())
        targets = new_targets(archive, curriculum, args.demos - active)
        build_demos(ctx, args, archive, targets, curriculum)
        if not any(not d["solved"] for d in curriculum.demos.values()):
            # Au plus N cellules par salle (groupe, salle) : chaque salle pèse à peu près autant.
            by_room = {}
            for c in sorted(glob.glob(os.path.join(args.archive, "*.state.z"))):
                by_room.setdefault(tuple(os.path.basename(c).split("-")[:2]), []).append(c)
            rng = random.Random(0)
            curriculum.cells = [c for cs in by_room.values()
                                for c in rng.sample(cs, min(len(cs), args.cells_per_room))]
            print(f"aucune démo à suivre dans {args.archive} : départs pris au hasard dans "
                  f"{len(curriculum.cells)} cellules de {len(by_room)} salles, sans recul vers Horon")
        else:
            curriculum.cells = []
        return bool(targets)

    demo_data = [None]

    def imitate(n_steps):
        """Recharge les démos et fait n_steps pas d'imitation pure."""
        demo_data[0] = DemoData([os.path.join(args.run, "demos", n) for n in curriculum.demos])
        if not len(demo_data[0]) or not n_steps:
            return
        model.train()
        total = 0.0
        for _ in range(n_steps):
            loss = bc_loss(model, demo_data[0], args.bc_batch, args.bc_len, device)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total += loss.item()
        print(f"imitation : {n_steps} pas sur {len(demo_data[0])} pas de démos, perte {total / n_steps:.3f}")

    if refresh() or not args.resume:
        imitate(args.bc_burst)
    else:
        imitate(0)
    print(f"PPO sur {device}, {args.workers} workers x {args.envs_per_worker} émulateurs, "
          f"démos : {curriculum.summary() or 'aucune'}")

    max_steps = int(args.max_seconds * 60 / FRAMES_PER_ACTION)
    patience = int(args.patience * 60 / FRAMES_PER_ACTION)
    vec = VecEnv(ctx, args.workers, args.envs_per_worker, args.rom, args.start,
                 os.path.join(args.run, "demos"), max_steps, patience)
    n_env, T = args.workers * args.envs_per_worker, args.steps
    frame_buf = torch.zeros((T, n_env) + SCREEN, dtype=torch.uint8, device=device)
    vec_buf = torch.zeros((T, n_env, PPO_INPUTS), device=device)
    act_buf = torch.zeros((T, n_env), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, n_env), device=device)
    rew_buf = torch.zeros((T, n_env), device=device)
    done_buf = torch.zeros((T, n_env), device=device)
    val_buf = torch.zeros((T, n_env), device=device)

    def save():
        save_model(model, model_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "curriculum": curriculum.state(), "kind": "cnn-lstm", "n_vec": PPO_INPUTS, "arch": args.arch},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    to = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    frames, vecs = map(to, vec.reset(curriculum.message()))
    done = torch.zeros(n_env, device=device)
    lstm_state = model.initial_state(n_env, device)
    cur_msg = None
    eps, horon_eps = [], []
    t_log = t_save = t_solo = t_refresh = time.time()
    t_solo -= args.solo_minutes * 60 - 120    # première partie solo après 2 min
    steps_log, update, solo_proc = step, 0, None
    bc_val = 0.0
    try:
        while args.total_steps is None or step < args.total_steps:
            start_state = (lstm_state[0].clone(), lstm_state[1].clone())
            model.eval()
            for t in range(T):
                with torch.no_grad():
                    logits, v, lstm_state = model(frames[None], vecs[None], lstm_state, done[None])
                    dist = Categorical(logits=logits[0])
                    a = dist.sample()
                frame_buf[t], vec_buf[t], done_buf[t] = frames, vecs, done
                act_buf[t], logp_buf[t], val_buf[t] = a, dist.log_prob(a), v[0]
                f, o, r, d, infos = vec.step(a.cpu().numpy(), cur_msg)
                cur_msg = None
                frames, vecs, rew_buf[t], done = to(f), to(o), to(r), to(d)
                for info in infos:
                    (horon_eps if info["demo"] is None and info["target"] is None else eps).append(info)
                    event = curriculum.record(info)
                    if event:
                        cur_msg = curriculum.message()
                        print(f"démo {info['demo']} {event} : {curriculum.summary()}")
                        if event == "résolue":
                            # Démo résolue depuis Horon : nouvelle cible, archive relue car elle grandit.
                            if refresh():
                                imitate(args.bc_burst)
                            cur_msg = curriculum.message()
            step += T * n_env

            with torch.no_grad():
                _, next_v, _ = model(frames[None], vecs[None], lstm_state, done[None])
                next_v = next_v[0]
            adv = torch.zeros_like(rew_buf)
            last = torch.zeros(n_env, device=device)
            for t in reversed(range(T)):
                # done_buf[t] dit si l'observation t ouvre un nouvel épisode ; done celle d'après.
                nonterm = 1.0 - (done if t == T - 1 else done_buf[t + 1])
                nv = next_v if t == T - 1 else val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nv * nonterm - val_buf[t]
                last = delta + args.gamma * args.lam * nonterm * last
                adv[t] = last
            ret = adv + val_buf

            # Avec le LSTM, un minibatch = des envs entiers sur les T pas, rejoués depuis leur mémoire de départ.
            model.train()
            per_mb = max(1, n_env // args.minibatches)
            for _ in range(args.epochs):
                perm = torch.randperm(n_env, device=device)
                for i in range(0, n_env, per_mb):
                    e = perm[i:i + per_mb]
                    logits, v, _ = model(frame_buf[:, e], vec_buf[:, e],
                                         (start_state[0][:, e], start_state[1][:, e]), done_buf[:, e])
                    dist = Categorical(logits=logits)
                    ratio = (dist.log_prob(act_buf[:, e]) - logp_buf[:, e]).exp()
                    a_n = adv[:, e]
                    a_n = (a_n - a_n.mean()) / (a_n.std() + 1e-8)
                    pg = torch.max(-a_n * ratio, -a_n * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    vl = 0.5 * (v - ret[:, e]).pow(2).mean()
                    ent = dist.entropy().mean()
                    loss = pg + args.vf * vl - args.ent * ent
                    # Imitation une fois par époque : elle coûte autant qu'un minibatch PPO.
                    if i == 0 and args.bc_coef and demo_data[0] is not None and len(demo_data[0]):
                        bc = bc_loss(model, demo_data[0], args.bc_batch, args.bc_len, device)
                        if bc is not None:
                            loss = loss + args.bc_coef * bc
                            bc_val = bc.item()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
            update += 1

            now = time.time()
            if now - t_log >= 60:
                sps = (step - steps_log) / (now - t_log)
                h = (f"horon : score moyen {np.mean([e['score'] for e in horon_eps]):.1f}, "
                     f"max {max(e['score'] for e in horon_eps)}, kills {np.mean([e['kills'] for e in horon_eps]):.1f} "
                     f"({len(horon_eps)} ép.)" if horon_eps else "horon : -")
                g = (f"démos : succès {100 * np.mean([e['success'] for e in eps]):.0f} % "
                     f"({len(eps)} ép.)" if eps else "démos : -")
                print(f"ppo : étape {step / 1e6:.2f} M, {sps:.0f} pas/s | {h} | {g} | "
                      f"pi {pg.item():.3f} v {vl.item():.3f} entropie {ent.item():.2f} imitation {bc_val:.2f}")
                if curriculum.demos:
                    print(f"curriculum : {curriculum.summary()}")
                eps, horon_eps, t_log, steps_log = [], [], now, step
            if now - t_refresh >= args.refresh_minutes * 60:
                if refresh():
                    imitate(args.bc_burst)
                    print(f"curriculum : {curriculum.summary()}")
                cur_msg = curriculum.message()
                t_refresh = now
            if now - t_save >= 300:
                save()
                t_save = now
            if now - t_solo >= args.solo_minutes * 60 and (solo_proc is None or not solo_proc.is_alive()):
                save()
                solo_proc = ctx.Process(target=solo_eval, daemon=True,
                                        args=(args.rom, args.start, model_path, args.solo_seconds, step, args.run))
                solo_proc.start()
                t_solo = now
    except KeyboardInterrupt:
        pass
    save()
    print(f"arrêt à l'étape {step}, modèle dans {model_path}")
    vec.close()
    if solo_proc is not None and solo_proc.is_alive():
        solo_proc.terminate()


if __name__ == "__main__":
    main()
