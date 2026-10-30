"""Fabrique un state "fin de jeu" : prend un state existant et remplit l'inventaire en RAM.
Adresses : oracles-disasm include/wram.s (valeurs Seasons), trésors : constants/common/treasure.s."""
import argparse

from zneat.archive import read_state
from zneat.env import ZeldaEnv

SWORD, SHIELD, BOMBS, BOOMERANG, ROD, MAGNET = 0x05, 0x01, 0x03, 0x06, 0x07, 0x08
SLINGSHOT, SHOVEL, BRACELET, FEATHER, SATCHEL = 0x13, 0x15, 0x16, 0x17, 0x19
SEEDS = (0x20, 0x21, 0x22, 0x23, 0x24)
FLIPPERS = 0x2E

TREASURES = (SWORD, SHIELD, BOMBS, BOOMERANG, ROD, MAGNET, SLINGSHOT, SHOVEL,
             BRACELET, FEATHER, SATCHEL, FLIPPERS) + SEEDS

RAM = {
    0xC6A3: 0x10, 0xC6A2: 0x10,          # 4 coeurs (en quarts)
    0xC6A5: 0x99, 0xC6A6: 0x09,          # 999 rubis (BCD)
    0xC6A9: 3,                           # bouclier miroir
    0xC6AA: 0x50, 0xC6AB: 0x50,          # bombes 50/50 (BCD)
    0xC6AC: 3,                           # épée de maître
    0xC6AE: 3,                           # sacoche niveau 3
    0xC6B0: 0x0F,                        # les 4 saisons du Rod
    0xC6B1: 2,                           # boomerang magique
    0xC6B3: 2,                           # hyper fronde
    0xC6B4: 2,                           # cape de Roc
    0xC6B5: 0x50, 0xC6B6: 0x50, 0xC6B7: 0x50, 0xC6B8: 0x50, 0xC6B9: 0x50,  # graines (BCD)
    0xC6BB: 0xFF,                        # les 8 essences
}

ap = argparse.ArgumentParser()
ap.add_argument("--rom", default="rom/oos.gbc")
ap.add_argument("--from", dest="src", required=True, help="state de départ (.state ou .state.z)")
ap.add_argument("--out", default="states/horon-endgame.state")
ap.add_argument("--b", type=lambda x: int(x, 0), default=SWORD, help="objet sur B")
ap.add_argument("--a", type=lambda x: int(x, 0), default=FEATHER, help="objet sur A")
args = ap.parse_args()

env = ZeldaEnv(args.rom)
env.load(read_state(args.src))
m = env.pb.memory
for addr, v in RAM.items():
    m[addr] = v
for t in TREASURES:
    m[0xC692 + t // 8] |= 1 << (t % 8)
items = [SWORD, ROD, FEATHER, SHIELD, BOMBS, BOOMERANG, MAGNET, SLINGSHOT, SHOVEL, BRACELET, SATCHEL]
m[0xC680], m[0xC681] = args.b, args.a
storage = [i for i in items if i not in (args.b, args.a)]
for i in range(16):
    m[0xC682 + i] = storage[i] if i < len(storage) else 0
env.step(set(), 2)
open(args.out, "wb").write(env.save())
print("écrit", args.out)
