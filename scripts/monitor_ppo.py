#!/usr/bin/env python3
"""Moniteur d'entraînement PPO :
Suit l'avancement de runs/ppo-impala, évalue le modèle tous les 4M de pas,
compare les métriques avec le checkpoint de référence à 30M, et consigne
les résultats dans runs/ppo-impala/comparatif_milestones.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/mnt/data/dev/zelda-neat")
import torch
from collections import Counter
from zneat import ram
from zneat.archive import read_state
from zneat.env import ZeldaEnv, EQUIP_A
from zneat.obs import ACTIONS, Observer, observe
from zneat.ppo_model import load_model
from zneat.runner import FRAMES_PER_ACTION

RUN_DIR = "runs/ppo-impala"
CKPT_PATH = os.path.join(RUN_DIR, "ckpt.pt")
MODEL_PATH = os.path.join(RUN_DIR, "model.pt")
REPORT_PATH = os.path.join(RUN_DIR, "comparatif_milestones.json")
BASELINE_STEP = 30220288
STEP_INTERVAL = 5_000_000
MAX_STEP = 50_000_000

BASELINE_METRICS = {
    "step": BASELINE_STEP,
    "avg_shield_pct": 21.5,
    "avg_dir_switches": 551.3,
    "avg_dialogue_starts": 16.7,
    "avg_rooms": 3.7
}

def evaluate_model(model_path, num_steps=1200, seeds=(0, 42, 123)):
    device = "cpu"
    try:
        model = load_model(model_path).to(device)
    except Exception as e:
        print(f"Erreur chargement modèle {model_path}: {e}", flush=True)
        return None
    model.eval()

    from torch.distributions import Categorical

    results = []
    for seed in seeds:
        env = ZeldaEnv("rom/oos.gbc")
        env.load(read_state("states/horon-endgame.state"))
        m = env.pb.memory
        m[ram.LINK_MAX_HEALTH] = min(env.mem[ram.LINK_MAX_HEALTH], 16)
        m[ram.LINK_HEALTH] = min(env.mem[ram.LINK_HEALTH], 16)
        m[ram.RUPEES] = 0
        m[ram.RUPEES + 1] = 0
        m[0xCBEA] = 0x09
        env.step(set(), 1, True)

        observer = Observer()
        obs = observe(env, observer)
        lstm_state = model.initial_state(1, device)
        torch.manual_seed(seed)

        shield_steps = 0
        dir_switches = 0
        last_dir = None
        dir_streak = 0
        dialogues = 0
        prev_text = 0
        rooms_visited = set()
        rooms_visited.add((m[ram.ACTIVE_GROUP], m[ram.ACTIVE_ROOM]))
        action_counts = Counter()

        for step in range(num_steps):
            f = torch.from_numpy(obs[0][None, None]).to(device)
            v = torch.from_numpy(obs[1][None, None]).to(device)
            done_t = torch.zeros((1, 1), device=device)
            with torch.no_grad():
                logits, val, lstm_state = model(f, v, lstm_state, done_t)
                dist = Categorical(logits=logits[0, 0])
                a = dist.sample().item()

            act = ACTIONS[a]
            action_counts[tuple(act)] += 1
            if "a" in act and env.mem[EQUIP_A] == 0x01:
                shield_steps += 1

            move_dirs = [d for d in ("up", "down", "left", "right") if d in act]
            if move_dirs:
                cur_dir = move_dirs[0]
                if last_dir is not None and cur_dir != last_dir:
                    if dir_streak < 2:
                        dir_switches += 1
                    dir_streak = 0
                else:
                    dir_streak += 1
                last_dir = cur_dir

            env.step(set(act), FRAMES_PER_ACTION, True)
            mem = env.mem
            room = (mem[ram.ACTIVE_GROUP], mem[ram.ACTIVE_ROOM])
            rooms_visited.add(room)

            text_act = mem[ram.TEXT_ACTIVE]
            if text_act and not prev_text:
                dialogues += 1
            prev_text = text_act

            obs = observe(env, observer)

        top_act, top_cnt = action_counts.most_common(1)[0]
        results.append({
            "seed": seed,
            "shield_pct": round(shield_steps / num_steps * 100, 1),
            "dir_switches": dir_switches,
            "dialogue_starts": dialogues,
            "rooms_visited": len(rooms_visited),
            "top_action": str(list(top_act)),
            "top_action_pct": round(top_cnt / num_steps * 100, 1)
        })

    avg_shield = sum(r["shield_pct"] for r in results) / len(results)
    avg_switches = sum(r["dir_switches"] for r in results) / len(results)
    avg_dialogues = sum(r["dialogue_starts"] for r in results) / len(results)
    avg_rooms = sum(r["rooms_visited"] for r in results) / len(results)

    return {
        "avg_shield_pct": round(avg_shield, 1),
        "avg_dir_switches": round(avg_switches, 1),
        "avg_dialogue_starts": round(avg_dialogues, 1),
        "avg_rooms": round(avg_rooms, 1),
        "runs": results
    }

def get_current_step():
    if not os.path.exists(CKPT_PATH):
        return None
    try:
        ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        return ck.get("step")
    except Exception:
        return None

def main():
    print(f"[MONITOR] Démarrage du moniteur pour {RUN_DIR}", flush=True)
    next_milestone = STEP_INTERVAL

    history = []
    if os.path.exists(REPORT_PATH):
        try:
            data = json.load(open(REPORT_PATH))
            history = data.get("milestones", [])
            if history:
                last_step = history[-1]["step"]
                while next_milestone <= last_step:
                    next_milestone += STEP_INTERVAL
        except Exception:
            history = []

    print(f"[MONITOR] Prochain jalon comparatif : {next_milestone / 1e6:.1f}M pas", flush=True)

    while True:
        step = get_current_step()
        if step is not None:
            if step >= next_milestone:
                print(f"\n[MONITOR] === Jalon atteint : {step} pas (objectif {next_milestone}) ===", flush=True)
                metrics = evaluate_model(MODEL_PATH)
                if metrics:
                    entry = {
                        "milestone": f"{step / 1e6:.1f}M",
                        "step": step,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "metrics": metrics,
                        "comparison_vs_baseline": {
                            "shield_pct_diff": round(metrics["avg_shield_pct"] - BASELINE_METRICS["avg_shield_pct"], 1),
                            "dir_switches_diff": round(metrics["avg_dir_switches"] - BASELINE_METRICS["avg_dir_switches"], 1),
                            "dialogue_starts_diff": round(metrics["avg_dialogue_starts"] - BASELINE_METRICS["avg_dialogue_starts"], 1),
                            "rooms_diff": round(metrics["avg_rooms"] - BASELINE_METRICS["avg_rooms"], 1)
                        }
                    }
                    history.append(entry)
                    report = {
                        "baseline_30m": BASELINE_METRICS,
                        "milestones": history
                    }
                    with open(REPORT_PATH, "w") as f:
                        json.dump(report, f, indent=2)
                    print(f"[MONITOR] Comparatif enregistré dans {REPORT_PATH}", flush=True)
                    print(f"[MONITOR] Shield: {metrics['avg_shield_pct']}% (base: {BASELINE_METRICS['avg_shield_pct']}%) | "
                          f"Dialogues: {metrics['avg_dialogue_starts']} (base: {BASELINE_METRICS['avg_dialogue_starts']}) | "
                          f"Dir switches: {metrics['avg_dir_switches']} (base: {BASELINE_METRICS['avg_dir_switches']})", flush=True)

                next_milestone += STEP_INTERVAL

            if step >= MAX_STEP:
                print(f"[MONITOR] Objectif de {MAX_STEP / 1e6:.1f}M atteint !", flush=True)
                break

        time.sleep(30)

if __name__ == "__main__":
    main()
