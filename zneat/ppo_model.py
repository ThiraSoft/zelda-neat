"""Réseau du joueur (CNN sur l'écran + vecteur, puis LSTM) et partie solo d'un modèle sauvé."""
import json
import os
import shutil
import signal

import numpy as np
import torch
from torch import nn

from .archive import read_state
from .env import SCREEN, ZeldaEnv
from .obs import ACTIONS, PPO_INPUTS, Observer, observe
from .runner import solo, solo_line, solo_score


class Conv(nn.Conv2d):
    """Convolution par im2col + matmul. Sur ROCm (RX 9070), les convolutions MIOpen
    sont 4 fois plus lentes que ce passage par rocBLAS. Mêmes poids que nn.Conv2d."""

    def forward(self, x):
        p = self.padding[0]
        if p:
            x = nn.functional.pad(x, (p, p, p, p))
        B, _, H, W = x.shape
        k, s = self.kernel_size[0], self.stride[0]
        cols = nn.functional.unfold(x, k, stride=s)
        y = self.weight.view(self.out_channels, -1) @ cols + self.bias.view(1, -1, 1)
        return y.view(B, self.out_channels, (H - k) // s + 1, (W - k) // s + 1)


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.ReLU(), Conv(c, c, 3, 1, 1), nn.ReLU(), Conv(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.body(x)


def impala_cnn(channels=(32, 64, 64)):
    """CNN résiduel d'IMPALA, plus profond que celui de DQN : 3 étages conv + max pool + 2 blocs."""
    layers, c_in = [], 1
    for c in channels:
        layers += [Conv(c_in, c, 3, 1, 1), nn.MaxPool2d(3, 2, 1), Residual(c), Residual(c)]
        c_in = c
    return nn.Sequential(*layers, nn.ReLU(), nn.Flatten())


class Policy(nn.Module):
    """arch "nature" : CNN de DQN, fc 256. arch "impala" : CNN élargi (48->96->192), fc 512 (5,6M params)."""

    def __init__(self, n_vec=PPO_INPUTS, n_act=len(ACTIONS), hidden=256, arch="nature"):
        super().__init__()
        self.hidden, self.n_vec, self.arch = hidden, n_vec, arch
        if arch == "impala":
            self.cnn = nn.Sequential(
                Conv(1, 48, 8, 4), nn.ReLU(),
                Conv(48, 96, 4, 2), nn.ReLU(),
                Conv(96, 192, 3, 1), nn.ReLU(),
                nn.Flatten())
            n_fc = 512
        else:
            self.cnn = nn.Sequential(
                Conv(1, 32, 8, 4), nn.ReLU(), Conv(32, 64, 4, 2), nn.ReLU(),
                Conv(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
            n_fc = 256
        with torch.no_grad():
            n_cnn = self.cnn(torch.zeros(1, 1, *SCREEN)).shape[1]
        self.cnn_fc = nn.Sequential(nn.Linear(n_cnn, n_fc), nn.ReLU())
        self.vec = nn.Sequential(nn.Linear(n_vec, 128), nn.ReLU())
        self.lstm = nn.LSTM(n_fc + 128, hidden)
        self.pi = nn.Linear(hidden, n_act)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def initial_state(self, n, device="cpu"):
        z = torch.zeros(1, n, self.hidden, device=device)
        return z, z.clone()

    def forward(self, frames, vec, state, done):
        """frames (T, B, 72, 80) uint8, vec (T, B, n_vec), done (T, B) : 1 si le pas ouvre un épisode,
        la mémoire est alors remise à zéro. Renvoie logits (T, B, A), valeurs (T, B), état."""
        T, B = frames.shape[:2]
        x = self.cnn(frames.reshape(T * B, 1, *SCREEN).float() / 255)
        x = torch.cat((self.cnn_fc(x), self.vec(vec.reshape(T * B, -1).float())), -1).reshape(T, B, -1)
        # Le LSTM tourne d'un bloc entre deux débuts d'épisode (rares) : bien plus rapide qu'un pas à la fois.
        starts = [0] + [t for t in done[1:].any(1).nonzero().flatten().add(1).tolist()] + [T]
        out = []
        h, c = state
        for s, e in zip(starts, starts[1:]):
            keep = (1.0 - done[s]).view(1, B, 1)
            y, (h, c) = self.lstm(x[s:e], (h * keep, c * keep))
            out.append(y)
        y = torch.cat(out)
        return self.pi(y), self.v(y).squeeze(-1), (h, c)


def save_model(model, path):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "hidden": model.hidden, "n_vec": model.n_vec,
                "kind": "cnn-lstm", "arch": model.arch}, tmp)
    os.replace(tmp, path)


def load_model(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("kind") != "cnn-lstm":
        raise SystemExit(f"{path} : modèle non reconnu")
    n_vec = data.get("n_vec", PPO_INPUTS)
    n_act = data["model"]["pi.weight"].shape[0] if "model" in data and "pi.weight" in data["model"] else len(ACTIONS)
    model = Policy(n_vec=n_vec, n_act=n_act, hidden=data["hidden"], arch=data.get("arch", "nature"))
    model.load_state_dict(data["model"])
    return model.eval()


def torch_policy(model, greedy=False, seed=0):
    """policy(env) -> boutons, pour runner.rollout avec render=True (l'écran doit être rendu).
    Une politique par partie : elle garde la mémoire du LSTM et les salles vues."""
    gen = torch.Generator().manual_seed(seed)
    observer = Observer()
    state = [model.initial_state(1)]
    zero = torch.zeros(1, 1)
    held_a = [False]
    held_b = [False]
    cooldown_a = [0]
    cooldown_b = [0]

    @torch.no_grad()
    def policy(env):
        f, v = observe(env, observer)
        if len(v) != model.n_vec:
            if len(v) > model.n_vec:
                v = v[:model.n_vec]
            else:
                v = np.pad(v, (0, model.n_vec - len(v)))
        logits, _, state[0] = model(torch.from_numpy(f)[None, None], torch.from_numpy(v)[None, None],
                                    state[0], zero)
        logits = logits[0, 0]
        if greedy:
            act = ACTIONS[int(logits.argmax())]
        else:
            act = ACTIONS[int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))]

        wanted = set(act)
        if "a" in wanted:
            if held_a[0]:
                pass
            elif cooldown_a[0] > 0:
                wanted.remove("a")
                cooldown_a[0] -= 1
            else:
                held_a[0] = True
        else:
            if held_a[0]:
                held_a[0] = False
                cooldown_a[0] = 2
            elif cooldown_a[0] > 0:
                cooldown_a[0] -= 1

        if "b" in wanted:
            if held_b[0]:
                pass
            elif cooldown_b[0] > 0:
                wanted.remove("b")
                cooldown_b[0] -= 1
            else:
                held_b[0] = True
        else:
            if held_b[0]:
                held_b[0] = False
                cooldown_b[0] = 2
            elif cooldown_b[0] > 0:
                cooldown_b[0] -= 1

        return wanted
    return policy


def solo_eval(rom, seed_path, model_path, seconds, step, run):
    """Joue le modèle seul depuis Horon, échantillonné puis glouton. Garde dans model-best.pt
    le modèle au meilleur score échantillonné (c'est lui qui servira à générer des parties)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    model = load_model(model_path)
    env = ZeldaEnv(rom)
    state = read_state(seed_path)
    ts = solo(env, torch_policy(model, seed=0), state, seconds, render=True)
    tg = solo(env, torch_policy(model, greedy=True), state, seconds, render=True)
    best_path = os.path.join(run, "solo-best.json")
    best = json.load(open(best_path)) if os.path.exists(best_path) else {"score": -1}
    score = solo_score(ts)
    mark = ""
    if score > best["score"]:
        shutil.copy(model_path, os.path.join(run, "model-best.pt"))
        json.dump({"score": score, "step": step}, open(best_path, "w"))
        mark = ", nouveau meilleur"
    print(f"solo : {solo_line(ts)} [échantillonné] | glouton score {solo_score(tg)} "
          f"(étape {step / 1e6:.2f} M{mark})", flush=True)


class DemoData:
    """Fenêtres de démos pour l'imitation : (frames, vec, act) de longueur L, act = -1 à ignorer."""

    def __init__(self, dirs, rng=None):
        self.rng = rng or np.random.default_rng()
        self.demos = []
        for d in dirs:
            try:
                arrs = [np.load(os.path.join(d, f), mmap_mode="r") for f in ("frames.npy", "vec.npy", "act.npy")]
            except FileNotFoundError:
                continue
            if len(arrs[2]):
                self.demos.append(arrs)
        self.sizes = np.array([len(a[2]) for a in self.demos], dtype=np.float64)

    def __len__(self):
        return int(self.sizes.sum())

    def batch(self, n, L, device):
        frames = np.zeros((L, n) + SCREEN, np.uint8)
        vec = np.zeros((L, n, PPO_INPUTS), np.float32)
        act = np.full((L, n), -1, np.int64)
        p = self.sizes / self.sizes.sum()
        for j, i in enumerate(self.rng.choice(len(self.demos), n, p=p)):
            f, v, a = self.demos[i]
            s = int(self.rng.integers(0, max(1, len(a) - L + 1)))
            k = min(L, len(a) - s)
            frames[:k, j], vec[:k, j], act[:k, j] = f[s:s + k], v[s:s + k], a[s:s + k]
        return (torch.from_numpy(frames).to(device), torch.from_numpy(vec).to(device),
                torch.from_numpy(act).to(device))


def bc_loss(model, data, n, L, device):
    """Imitation : entropie croisée sur les actions des démos (fenêtres de L pas, mémoire à zéro)."""
    frames, vec, act = data.batch(n, L, device)
    done = torch.zeros(L, n, device=device)
    logits, _, _ = model(frames, vec, model.initial_state(n, device), done)
    mask = act >= 0
    if not mask.any():
        return None
    return nn.functional.cross_entropy(logits[mask], act[mask])
