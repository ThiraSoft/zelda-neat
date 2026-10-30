# Adresses WRAM d'Oracle of Seasons (US), tirées de oracles-disasm (include/wram.s).
# Les commentaires du désassemblage donnent "Ages/Seasons" : on garde la seconde valeur.

ACTIVE_GROUP = 0xCC49     # 0 overworld, 1 Subrosia, 4/5 donjons...
ACTIVE_ROOM = 0xCC4C
SEASON = 0xCC4E           # wRoomStateModifier : 0 printemps, 1 été, 2 automne, 3 hiver
TEXT_ACTIVE = 0xCBA0      # vérifié sur la ROM EU : 1 pendant un dialogue
OPENED_MENU = 0xCBCB      # wOpenedMenuType : 0 en jeu, 1 inventaire, 2 carte...
DISABLED_OBJECTS = 0xCCA4 # bit 0 ou 7 : Link bloqué (cinématique), non vérifié sur la ROM EU

LINK_HEALTH = 0xC6A2      # en quarts de coeur
LINK_MAX_HEALTH = 0xC6A3
RUPEES = 0xC6A5           # 2 octets en BCD (0xC6A5 et 0xC6A6, little-endian)

# Octets dont les bits comptent comme progression d'histoire.
TREASURE_FLAGS = (0xC692, 16)
ESSENCES = (0xC6BB, 1)
GLOBAL_FLAGS = (0xC6CA, 16)
STORY_RANGES = (TREASURE_FLAGS, ESSENCES, GLOBAL_FLAGS)

# Objets : 16 pages $d000-$df00, chaque page contient item/interaction/ennemi/part.
LINK = 0xD000
INTERACTION_BASE = 0xD040  # PNJ, mais aussi des déclencheurs invisibles
ENEMY_BASE = 0xD080
PART_BASE = 0xD0C0
OBJ_STRIDE = 0x100
NUM_SLOTS = 16

OBJ_ENABLED = 0x00
OBJ_ID = 0x01
OBJ_DIRECTION = 0x08
OBJ_YH = 0x0B
OBJ_XH = 0x0D
OBJ_VISIBLE = 0x1A        # bit 7 : visible
OBJ_HEALTH = 0x29

# Carte de collision de la salle : lignes de 16 octets, 12 lignes max.
ROOM_COLLISIONS = 0xCE00
ROOM_STRIDE = 16
ROOM_ROWS = 12
ROOM_IS_LARGE = 0xCC4A    # 0 : 10x8 cases, 1 : 15x11
