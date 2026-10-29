"""Mesure le débit d'évaluation selon le nombre de workers, pour choisir --workers."""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import neat

from zneat import runner
from zneat.env import BUTTONS


def bench_genome(args):
    genome, config, frames = args
    net = runner.make_net(genome, config)
    env, state = runner._env, runner._states[0]
    env.load(state)
    # Budget fixe, sans arrêt sur mort ou inactivité, pour comparer à charge égale.
    done = 0
    while done < frames:
        out = net.activate(env.observe())
        env.step({b for b, v in zip(BUTTONS, out) if v > 0.5}, runner.FRAMES_PER_ACTION)
        done += runner.FRAMES_PER_ACTION
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="rom/oos.gbc")
    ap.add_argument("--states", default="states/*.state")
    ap.add_argument("--config", default="config-neat.ini")
    ap.add_argument("--workers", default=None, help="liste, ex. 1,2,4,8,12")
    ap.add_argument("--genomes", type=int, default=64)
    ap.add_argument("--frames", type=int, default=1800, help="frames émulées par génome")
    args = ap.parse_args()

    states = sorted(glob.glob(args.states))
    if not states:
        raise SystemExit(f"aucun save state dans {args.states}")
    cpus = os.cpu_count()
    counts = ([int(x) for x in args.workers.split(",")] if args.workers
              else sorted({1, 2, cpus // 2, cpus - 1, cpus, cpus + cpus // 2} - {0}))

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)
    genomes = list(neat.Population(config).population.values())[:args.genomes]
    jobs = [(g, config, args.frames) for g in genomes]

    print(f"{len(jobs)} génomes x {args.frames} frames, {cpus} CPU")
    print(f"{'workers':>8} {'génomes/s':>10} {'frames/s':>10} {'x temps réel':>13} {'gain':>6}")
    base = None
    for n in counts:
        with Pool(n, initializer=runner.init_worker, initargs=(args.rom, states[:1])) as pool:
            pool.map(bench_genome, jobs[:n])  # chauffe
            t = time.perf_counter()
            frames = sum(pool.map(bench_genome, jobs, chunksize=1))
            dt = time.perf_counter() - t
        fps = frames / dt
        base = base or fps / n
        print(f"{n:>8} {len(jobs) / dt:>10.1f} {fps:>10.0f} {fps / 60:>13.0f} {fps / base:>6.2f}")

    print("prends le plus petit nombre de workers après lequel frames/s ne monte plus vraiment")


if __name__ == "__main__":
    main()
