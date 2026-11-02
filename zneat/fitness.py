from . import ram

W_NOVEL = 200.0  # salle (groupe, salle) absente de toute l'archive Go-Explore
W_ROOM = 100.0   # salle (groupe, salle) visitée, la saison ne compte pas
W_VARIANT = 10.0 # même salle dans une autre saison : peu, sinon changer de saison sur place rapporte
W_TILE = 0.0    # les cases ne comptent pas, seules les salles comptent (elles servent à la patience)
W_ENEMY_KIND = 100.0  # autant qu'une salle : les types d'ennemis tués comptent dans le score
W_STORY = 100.0
W_DEATH = -20.0


def story_bits(m):
    return sum(bin(m[a + i]).count("1") for a, n in ram.STORY_RANGES for i in range(n))


_known_rooms = (None, frozenset())


def known_rooms(known):
    """Salles (groupe, salle) des clés de l'archive, gardées tant que l'archive ne change pas."""
    global _known_rooms
    if _known_rooms[0] is not known:
        _known_rooms = (known, frozenset(k[:2] for k in known))
    return _known_rooms[1]


class Tracker:
    """Suit une évaluation : salles (groupe, salle, saison), cases visitées, types d'ennemis tués, flags."""

    def __init__(self, m, known=frozenset(), cell_fn=None):
        self.known = known
        self.cell_fn = cell_fn
        self.cell = None
        self.novel = set()
        self.rooms = set()
        self.tiles = set()
        self.kinds = set()
        self.kills = 0
        self.story0 = story_bits(m)
        self.story = 0
        self.frames = 0
        self.dead = False
        self.last_room = None
        self.slots = [(0, 0)] * ram.NUM_SLOTS
        self.update(m, 0)

    def update(self, m, frames):
        """Renvoie True si quelque chose de nouveau a été obtenu."""
        self.frames += frames
        self.last_kills = []   # type de chaque ennemi tué pendant ce pas
        new = False

        room = (m[ram.ACTIVE_GROUP], m[ram.ACTIVE_ROOM], m[ram.SEASON])
        if room != self.last_room:
            # Les slots sont réutilisés d'une salle à l'autre.
            self.slots = [(0, 0)] * ram.NUM_SLOTS
            self.last_room = room
        if room not in self.rooms:
            self.rooms.add(room)
            new = True

        if self.cell_fn:
            self.cell = self.cell_fn(m)
            if self.cell not in self.known and self.cell not in self.novel:
                self.novel.add(self.cell)
                new = True

        tile = (room, m[ram.LINK + ram.OBJ_YH] >> 4, m[ram.LINK + ram.OBJ_XH] >> 4)
        if tile not in self.tiles:
            self.tiles.add(tile)
            new = True

        for s in range(ram.NUM_SLOTS):
            o = ram.ENEMY_BASE + s * ram.OBJ_STRIDE
            cur = (m[o + ram.OBJ_ID], m[o + ram.OBJ_HEALTH]) if m[o + ram.OBJ_ENABLED] else (0, 0)
            prev_id, prev_hp = self.slots[s]
            if prev_hp > 0 and cur[0] == prev_id and cur[1] == 0:
                self.kills += 1
                self.last_kills.append(prev_id)
                if prev_id not in self.kinds:
                    self.kinds.add(prev_id)
                    new = True
            self.slots[s] = cur

        story = story_bits(m) - self.story0
        if story > self.story:
            self.story = story
            new = True

        if m[ram.LINK_HEALTH] == 0:
            self.dead = True
        return new

    def fitness(self):
        maps = {r[:2] for r in self.rooms}
        novel_maps = {c[:2] for c in self.novel} - known_rooms(self.known)
        return (W_NOVEL * len(novel_maps)
                + W_ROOM * len(maps)
                + W_VARIANT * (len(self.rooms) - len(maps))
                + W_TILE * len(self.tiles)
                + W_ENEMY_KIND * len(self.kinds)
                + W_STORY * self.story
                + (W_DEATH if self.dead else 0.0))
