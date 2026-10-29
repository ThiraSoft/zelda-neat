import argparse
import copy
import glob
import json
import os
import pickle
import random
import zlib
from multiprocessing import Pool

import neat

from zneat.archive import Archive, cell_key, key_str, read_state
from zneat.env import NUM_INPUTS, ZeldaEnv
from zneat.runner import indexed_job, init_worker, job as run_job, solo_line

KEEP_CHECKPOINTS = 5
RESTART_FRESH = 0.3      # part de génomes neufs après un plateau
RESTART_PARENTS = 5      # derniers découvreurs du hall utilisés comme parents


class Prune(neat.reporting.BaseReporter):
    def __init__(self, prefix):
        self.prefix = prefix

    def end_generation(self, config, population, species_set):
        for old in checkpoints(self.prefix)[:-KEEP_CHECKPOINTS]:
            os.remove(old)


class Explorer:
    """Une génération = les génomes NEAT plus des parties aléatoires, depuis des cellules de
    l'archive. Les parties aléatoires, courtes, passent en dernier et occupent les cœurs
    pendant que les dernières parties NEAT finissent."""

    def __init__(self, pool, archive, run, starts, random_trials, random_frames,
                 solo_state=None, solo_every=5, solo_seconds=600):
        self.pool = pool
        self.solo_state = solo_state
        self.solo_every = solo_every
        self.solo_seconds = solo_seconds
        self.solo_pending = None
        self.archive = archive
        self.run = run
        self.starts = starts
        self.random_trials = random_trials
        self.random_frames = random_frames
        self.rng = random.Random()
        self.generation = 0
        self.hall = os.path.join(run, "hall")
        os.makedirs(self.hall, exist_ok=True)
        self.kills_path = os.path.join(run, "kills.json")
        self.kills = {"neat": 0, "random": 0, "kinds": []}
        if os.path.exists(self.kills_path):
            self.kills = json.load(open(self.kills_path))

    def evaluate(self, genomes, config):
        kp = self.archive.known_path
        starts = [(k, self.archive.path(k)) for k in self.archive.choose(self.starts, self.rng)]
        jobs = [("neat", kp, g, config, starts) for _, g in genomes]
        for k in self.archive.choose(self.random_trials, self.rng):
            jobs.append(("random", kp, (k, self.archive.path(k)), self.rng.getrandbits(32), self.random_frames))

        found = {"neat": 0, "random": 0}
        gen_kills = {"neat": 0, "random": 0}
        kinds = set(self.kills["kinds"])
        for i, (fit, captured, kills, killed) in self.pool.imap_unordered(indexed_job, enumerate(jobs)):
            kind = jobs[i][0]
            gen_kills[kind] += kills
            kinds |= killed
            if kind == "neat":
                genomes[i][1].fitness = fit
            for k, (z, parent, acts) in captured.items():
                if self.archive.add(k, z, self.generation, kind, parent, acts):
                    found[kind] += 1
                    if kind == "neat":
                        self.save_discoverer(genomes[i][1], k)
        self.archive.save()
        for kind in gen_kills:
            self.kills[kind] += gen_kills[kind]
        self.kills["kinds"] = sorted(kinds)
        json.dump(self.kills, open(self.kills_path, "w"))
        top = max(k[3] for k in self.archive.cells)
        print(f"archive : {len(self.archive)} cellules, {len(self.archive.rooms())} salles, "
              f"histoire max {top} (+{found['neat']} NEAT, +{found['random']} aléatoire)")
        print(f"kills : {gen_kills['neat']} NEAT, {gen_kills['random']} aléatoire cette génération, "
              f"{self.kills['neat'] + self.kills['random']} au total, "
              f"{len(kinds)} types {[hex(k) for k in sorted(kinds)]}")
        self.solo(genomes, config)
        self.generation += 1
        return found["neat"]

    def solo(self, genomes, config):
        """Toutes les solo_every générations, le meilleur génome joue seul depuis le départ,
        en tâche de fond dans le pool. Le résultat est affiché à la génération où il arrive."""
        if self.solo_pending and self.solo_pending[0].ready():
            res, gen, fit = self.solo_pending
            print(f"solo : {solo_line(res.get())} (génome de la génération {gen}, fitness {fit:.0f})")
            self.solo_pending = None
        if (self.solo_state and self.solo_every and not self.solo_pending
                and self.generation % self.solo_every == 0):
            best = max((g for _, g in genomes), key=lambda g: g.fitness)
            args = ("solo", None, best, config, self.solo_state, self.solo_seconds)
            self.solo_pending = (self.pool.apply_async(run_job, (args,)), self.generation, best.fitness)

    def save_discoverer(self, genome, key):
        name = f"gen{self.generation:05d}-{key_str(key)}.pkl"
        for path in (os.path.join(self.hall, name), os.path.join(self.run, "best.pkl")):
            with open(path, "wb") as f:
                pickle.dump(genome, f)
        json.dump({"cells": len(self.archive), "generation": self.generation, "file": name},
                  open(os.path.join(self.run, "best.json"), "w"))


def seed_archive(archive, rom, states):
    env = ZeldaEnv(rom)
    for p in states:
        data = read_state(p)
        env.load(data)
        env.step(set(), 1)
        k = cell_key(env.mem)
        if archive.add(k, zlib.compress(data, 1), -1, "seed"):
            print("archive : départ", key_str(k), "depuis", p)
    archive.save()


def restart(pop, run):
    """Nouvelle population : copies mutées des derniers découvreurs + génomes neufs."""
    config = pop.config
    hall = sorted(glob.glob(os.path.join(run, "hall", "*.pkl")))[-RESTART_PARENTS:]
    parents = [pickle.load(open(p, "rb")) for p in reversed(hall)]
    n_fresh = int(config.pop_size * RESTART_FRESH) if parents else config.pop_size
    new = pop.reproduction.create_new(config.genome_type, config.genome_config, n_fresh)
    i = 0
    while len(new) < config.pop_size:
        child = copy.deepcopy(parents[i % len(parents)])
        child.key = next(pop.reproduction.genome_indexer)
        child.fitness = None
        for _ in range(1 + i % 3):
            child.mutate(config.genome_config)
        new[child.key] = child
        i += 1
    pop.population = new
    pop.species = config.species_set_type(config.species_set_config, pop.reporters)
    pop.species.speciate(config, pop.population, pop.generation)
    print(f"plateau : nouvelle population ({len(parents)} parents, {n_fresh} neufs)")


def checkpoints(prefix):
    return sorted(glob.glob(prefix + "*"), key=lambda p: int(p[len(prefix):]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="rom/oos.gbc")
    ap.add_argument("--states", default="states/*.state", help="states de départ de l'archive")
    ap.add_argument("--config", default="config-neat.ini")
    ap.add_argument("--run", default="runs/horon")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--every", type=int, default=5, help="checkpoint toutes les N générations")
    ap.add_argument("--starts", type=int, default=3, help="cellules de départ NEAT tirées par génération")
    ap.add_argument("--random-trials", type=int, default=400,
                    help="parties aléatoires Go-Explore par génération (0 = NEAT seul)")
    ap.add_argument("--random-seconds", type=float, default=20, help="durée max d'une partie aléatoire")
    ap.add_argument("--max-seconds", type=float, default=120, help="durée max d'une partie (secondes de jeu)")
    ap.add_argument("--patience", type=float, default=15, help="arrêt après N secondes sans rien de nouveau")
    ap.add_argument("--plateau", type=int, default=50,
                    help="nouvelle population après N générations sans salle nouvelle (0 = jamais)")
    ap.add_argument("--solo-every", type=int, default=5,
                    help="le meilleur génome joue seul depuis le premier state toutes les N générations (0 = jamais)")
    ap.add_argument("--solo-seconds", type=float, default=600, help="durée de la partie solo")
    ap.add_argument("--resume", nargs="?", const="latest",
                    help="checkpoint à reprendre (sans valeur : le plus récent du run)")
    args = ap.parse_args()

    os.makedirs(args.run, exist_ok=True)
    prefix = os.path.join(args.run, "ckpt-")
    archive = Archive(os.path.join(args.run, "archive"))
    if not os.path.exists(archive.known_path):
        archive.save()
    if not len(archive):
        states = sorted(glob.glob(args.states))
        if not states:
            raise SystemExit(f"aucun save state dans {args.states} (lance make_state.py)")
        seed_archive(archive, args.rom, states)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)
    assert config.genome_config.num_inputs == NUM_INPUTS, NUM_INPUTS

    if args.resume:
        path = args.resume
        if path == "latest":
            found = checkpoints(prefix)
            if not found:
                raise SystemExit(f"aucun checkpoint dans {args.run}")
            path = found[-1]
        print("reprise depuis", path)
        pop = neat.Checkpointer.restore_checkpoint(path)
    else:
        if checkpoints(prefix):
            raise SystemExit(f"{args.run} contient déjà un entraînement : --resume ou autre --run")
        pop = neat.Population(config)

    ckpt = neat.Checkpointer(args.every, time_interval_seconds=900, filename_prefix=prefix)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(ckpt)
    pop.add_reporter(Prune(prefix))

    pool = Pool(args.workers, initializer=init_worker,
                initargs=(args.rom, (), args.max_seconds, args.patience))
    solo_state = (sorted(glob.glob(args.states)) or [None])[0]
    explorer = Explorer(pool, archive, args.run, args.starts, args.random_trials,
                        int(args.random_seconds * 60), solo_state, args.solo_every, args.solo_seconds)
    explorer.generation = pop.generation
    try:
        last_gen, done = pop.generation, 0
        while args.generations is None or done < args.generations:
            found = []
            pop.run(lambda g, c: found.append(explorer.evaluate(g, c)), 1)
            done += 1
            if found[0]:
                last_gen = pop.generation
            elif args.plateau and pop.generation - last_gen >= args.plateau:
                restart(pop, args.run)
                last_gen = pop.generation
    except KeyboardInterrupt:
        # La génération en cours n'est pas finie : on sauve la population telle qu'avant son évaluation.
        pool.terminate()
        ckpt.save_checkpoint(pop.config, pop.population, pop.species, pop.generation)
        return
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
