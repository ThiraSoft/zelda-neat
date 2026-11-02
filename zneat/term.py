import os
import shutil
import signal
import sys
import time

import numpy as np

from . import ram

def bcd_rupees(m):
    return ((m[ram.RUPEES + 1] >> 4) * 1000 + (m[ram.RUPEES + 1] & 0xF) * 100
            + (m[ram.RUPEES] >> 4) * 10 + (m[ram.RUPEES] & 0xF))


def frame_to_ansi(frame, target_w=40, pad_spaces=0):
    """Convertit ce que voit le modèle (env.frame(), (3, 72, 80) en moyenne 2x2) en texte ANSI truecolor
    avec demi-blocs ▀.
    - Si target_w >= 80 : 80x36 caractères, chaque demi-bloc est exactement 1 pixel vu par le modèle.
    - Sinon : 40x18 caractères pour mobile/neonmux sans défilement, moyenne 2x2 de plus sur l'image du modèle."""
    img = frame.transpose(1, 2, 0)
    if target_w < 80:
        s = img.astype(np.uint16)
        img = ((s[0::2, 0::2] + s[0::2, 1::2] + s[1::2, 0::2] + s[1::2, 1::2]) >> 2).astype(np.uint8)
    top, bottom = img[0::2], img[1::2]
    h = min(top.shape[0], bottom.shape[0])
    w = min(top.shape[1], bottom.shape[1])

    prefix = " " * pad_spaces if pad_spaces > 0 else ""
    out = ["\033[H"]  # Repositionne le curseur en haut à gauche sans effacer l'écran
    last_fg = None
    last_bg = None
    for y in range(h):
        line = [prefix]
        for x in range(w):
            fg = (int(top[y, x, 0]), int(top[y, x, 1]), int(top[y, x, 2]))
            bg = (int(bottom[y, x, 0]), int(bottom[y, x, 1]), int(bottom[y, x, 2]))
            if fg != last_fg or bg != last_bg:
                line.append(f"\033[38;2;{fg[0]};{fg[1]};{fg[2]};48;2;{bg[0]};{bg[1]};{bg[2]}m")
                last_fg = fg
                last_bg = bg
            line.append("▀")
        line.append("\033[0m\r\n")
        out.append("".join(line))
        last_fg = None
        last_bg = None
    return "".join(out)


class TermViewer:
    """Afficheur temps réel pour terminal ANSI truecolor zéro-scintillement."""

    def __init__(self, cols=0, fps=15.0, tag="Zelda PPO"):
        self.cols_setting = cols
        self.fps = fps
        self.tag = tag
        self.target_dt = (1.0 / fps) if fps > 0 else 0
        self.last_step_time = 0
        self._cleaned_up = False

        # Masquer curseur au démarrage (sans \033[2J pour éviter le flash noir)
        sys.stdout.write("\033[?25l\033[H")
        sys.stdout.flush()

        # Nettoyage en cas d'interruption
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        if not self._cleaned_up:
            self._cleaned_up = True
            # Restaurer curseur et couleur
            sys.stdout.write("\033[?25h\033[0m\r\n")
            sys.stdout.flush()

    def get_layout(self):
        term_cols, term_lines = shutil.get_terminal_size(fallback=(80, 24))
        if self.cols_setting > 0:
            target_w = self.cols_setting
        elif term_cols >= 80 and term_lines >= 38:
            target_w = 80
        else:
            target_w = 40
        pad = max(0, (term_cols - target_w) // 2)
        return target_w, pad, term_cols

    def render(self, env, t, buttons):
        target_w, pad, term_cols = self.get_layout()
        frame_str = frame_to_ansi(env.frame(), target_w=target_w, pad_spaces=pad)

        # HUD compact sur 1 seule ligne pour ne jamais dépasser la hauteur du terminal
        m = env.mem
        hp = m[ram.LINK_HEALTH]
        max_hp = m[ram.LINK_MAX_HEALTH]
        room = m[ram.ACTIVE_ROOM]
        grp = m[ram.ACTIVE_GROUP]
        rooms_visited = len({r[:2] for r in t.rooms})
        kills = len(t.kinds)
        rup = bcd_rupees(m)
        btn_str = " ".join(sorted(buttons)) if buttons else "-"
        time_s = t.frames / 60.0

        prefix = " " * pad if pad > 0 else ""
        grp_names = {0: "Overworld", 1: "Subrosia", 4: "Donjon", 5: "Donjon"}
        gname = grp_names.get(grp, f"G{grp}")
        hud = (f"{prefix}\033[1;33m[{self.tag}]\033[0m "
               f"\033[1;35m{gname}\033[0m:0x{room:02X} "
               f"Salles:\033[1;32m{rooms_visited}\033[0m "
               f"HP:{hp}/{max_hp} Rub:{rup} K:{kills} "
               f"{time_s:.0f}s \033[1;36m{btn_str}\033[0m")
        # Remplir le reste de la ligne avec des espaces pour écraser les résidus
        raw_hud_len = pad + len(f"[{self.tag}] {gname}:0x{room:02X} Salles:{rooms_visited} HP:{hp}/{max_hp} Rub:{rup} K:{kills} {time_s:.0f}s {btn_str}")
        fill = " " * max(0, term_cols - raw_hud_len - 1)
        hud_line = f"{hud}{fill}\033[0m"  # PAS de \r\n final pour ne pas déclencher de scrolling vertical

        sys.stdout.write(frame_str + hud_line)
        sys.stdout.flush()

        # Cadencement FPS
        if self.target_dt > 0:
            now = time.perf_counter()
            elapsed = now - self.last_step_time
            sleep_time = self.target_dt - elapsed
            if 0 < sleep_time < self.target_dt:
                time.sleep(sleep_time)
            self.last_step_time = time.perf_counter()
