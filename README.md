# zelda-neat

NEAT qui apprend à jouer à Zelda Oracle of Seasons dans PyBoy, sur CPU (un émulateur par cœur).

## Installer sur un autre PC

Prérequis : Linux (ou macOS), Python 3.12, et [uv](https://docs.astral.sh/uv/) de préférence. Tout tourne sur CPU, un worker par cœur.

    tar xzf zelda-neat.tar.gz && cd zelda-neat
    ./install.sh                      # crée .venv et installe requirements.txt
    .venv/bin/python bench_workers.py # trouve le bon --workers pour cette machine
    .venv/bin/python train.py --run runs/horon --states states/horon-endgame.state --resume --workers N

`--resume` reprend le run livré dans l'archive (checkpoint, archive Go-Explore, hall, kills). Sans `--resume`, il faut un autre `--run` pour repartir de zéro.

Pour laisser tourner en tâche de fond sous systemd :

    systemd-run --user --unit zelda-neat --working-directory=$PWD --setenv=PYTHONUNBUFFERED=1 \
      sh -c 'exec .venv/bin/python train.py --run runs/horon --states states/horon-endgame.state --resume >> runs/horon/train.log 2>&1'
    systemctl --user kill -s INT zelda-neat   # arrêt propre avec checkpoint

## Makefile

Toutes les commandes passent par `make` (`make help` les liste) :

    make install                 # crée .venv
    make bench                   # mesure le bon nombre de workers
    make start                   # explore par parties aléatoires en tâche de fond (Go-Explore, remplit l'archive)
    make neat                    # ancien entraînement NEAT, au premier plan
    make neat-spawns-start       # NEAT en score solo, départs au hasard dans spawns/ (run runs/neat-spawns)
    make neat-spawns-stop / neat-spawns-log
    make start-tmux              # pareil dans une session tmux "neat", sans systemd
    make run                     # pareil au premier plan, Ctrl+C sauve un checkpoint
    make stop                    # arrêt propre avec checkpoint (systemd ou tmux)
    make status                  # génération, salles, kills, plateaux, erreurs
    make log                     # suit la progression
    make replay                  # un génome NEAT (GENOME=..., plus fourni dans le dépôt) joue depuis Horon, en fenêtre
    make compare                 # NEAT et PPO jouent en même temps depuis Horon, deux fenêtres (COMPARE_SECONDS=120)
    make ppo-replay RANDOM_LOCATION=1   # départ dans une cellule de spawns/ au hasard (aussi replay, compare, ppo-solo)
    make reset CONFIRM=oui       # efface le run, le prochain start repart de zéro
    make package                 # archive code + ROM + states + run dans dist/ (la ROM n'est jamais dans git)
    make vault                   # package puis envoi dans le Televault
    make pull SERVER=user@host   # rapatrie le run du serveur (l'ancien local est renommé)

Joueur seul et PPO (détails plus bas) :

    make solo                    # le meilleur génome joue seul 10 min depuis Horon
    make paths                   # cellules de l'archive joignables depuis Horon par un chemin d'inputs
    make paths-check CHECK=50    # rejoue 50 chemins et vérifie qu'ils arrivent au bon endroit
    make ppo-start               # entraîne un seul réseau par PPO en tâche de fond
    make ppo-stop / ppo-log      # arrêt propre, suivi
    make ppo-solo                # le modèle PPO joue seul 10 min depuis Horon
    make ppo-replay              # pareil avec une fenêtre
    make install-ppo             # torch CPU dans .venv, pour une machine sans le venv ROCm
    make dataset-start           # enregistre des parties (écran + boutons) pour un world model

Variables (valeurs par défaut) : `WORKERS` (nproc), `RANDOM_TRIALS` (50 par worker), `RUN` (runs/horon), `STATES` (states/horon-endgame.state), `MAX_SECONDS` (120), `PATIENCE` (15), `RANDOM_SECONDS` (20), `GENERATIONS` (vide = sans limite), `ARGS` (options en plus pour train.py). Exemple pour des parties plus longues : `make stop && make start MAX_SECONDS=300 PATIENCE=30`.

Avec un run existant, la config NEAT vient du checkpoint, et les changements de `config-neat.ini` ne s'appliquent qu'à un nouveau run. Pour qu'un service systemd survive à la déconnexion sur un serveur, lance `loginctl enable-linger` une fois.

## État

ROM utilisée : version européenne (En, Fr, De, Es, It). `states/after-intro.state` : partie en anglais, juste après la scène de Din et d'Onox, avec Link libre en hiver au nord de Horon et sans épée. `states/intro/start.state` est le tout début, avant la troupe : pour lancer la scène, il faut parler aux 6 membres puis à Din. Sur cette version, c'est vérifié pour les PV, la salle, la position de Link et la grille de collision. La détection des ennemis tués a été vue une fois en action avec l'épée (type $09).

À savoir :
- Sans son émulé, le jeu boucle en attendant la puce son à chaque initialisation du canal 3 (au démarrage, mais aussi pendant certaines cinématiques). `zneat/env.py` neutralise ces deux sauts en mémoire (bank $39, $49d1 et $49ef) quand il tourne sans fenêtre. La partie est identique à celle avec son, et 40 % plus rapide.
- Les objets ($d000-$dfff) sont lus dans la banque WRAM 1, car le jeu change de banque en cours de frame.

Entraînement en tâche de fond : `systemctl --user status zelda-neat`, log dans `runs/horon/train.log`. Après un redémarrage : `.venv/bin/python train.py --resume`.

## Départ fin de jeu à Horon

`states/horon-endgame.state` : centre de Horon avec un inventaire de fin de jeu écrit en RAM par `make_endgame.py`. Il contient l'épée de maître (B), le Rod avec les 4 saisons (A), la cape de Roc, le bouclier miroir, le boomerang magique, l'hyper fronde, les gants magnétiques, la pelle, le bracelet, la sacoche et les 5 graines, les bombes, les palmes, les 8 essences, 16 cœurs et 999 rubis. Les flags d'histoire (événements, PNJ) ne sont pas modifiés. Le réseau n'a pas Start ni Select, il garde donc les objets placés sur A et B (`--a` et `--b` pour en changer).

    .venv/bin/python make_endgame.py --from <state> [--a 0x17]
    .venv/bin/python train.py --run runs/horon --states states/horon-endgame.state

## Démarrage rapide

1. Copier la ROM dans `rom/oos.gbc`.
2. `.venv/bin/python make_state.py` : jouer jusqu'à ce que Link soit contrôlable, puis fermer la fenêtre.
3. `.venv/bin/python probe.py` : vérifier que les salles, les PV et la grille d'obstacles collent avec l'écran.
4. `.venv/bin/python train.py`

Checkpoints :
- Toute la population est sauvée toutes les 5 générations et toutes les 15 minutes dans `runs/horon/ckpt-N`. Les 5 derniers sont gardés.
- `best.pkl` contient le dernier génome à avoir découvert une salle.
- Chaque génome découvreur est copié dans `hall/` et n'est jamais effacé.
- Ctrl+C sauve un checkpoint avant de quitter.

Options utiles :
- `--max-seconds 120` : durée max d'une partie, en secondes de jeu.
- `--patience 15` : la partie s'arrête après N secondes sans nouvelle case, ni nouvelle salle, ni nouvel ennemi tué, ni nouveau flag. Le compteur est en pause pendant les dialogues et les cinématiques. La durée max reste le garde-fou.
- `--plateau 50` : après N générations sans salle nouvelle, on repart d'une nouvelle population. 70 % sont des copies mutées des 5 derniers découvreurs du `hall/`, 30 % sont des génomes neufs. `0` désactive.
- `--starts 3` : nombre de salles de départ tirées dans l'archive à chaque génération.
- `--resume` reprend au dernier checkpoint, ou à celui qu'on donne en chemin.

## Préparer

1. Copier ta ROM dans `rom/oos.gbc`.
2. Créer un point de départ : `.venv/bin/python make_state.py`, jouer jusqu'à avoir Link contrôlable, fermer la fenêtre. Ça écrit `states/after-intro.state`. On peut en faire plusieurs (`--out states/xxx.state`), chaque génome est évalué sur tous et la fitness est la moyenne.
3. Vérifier les adresses RAM : `.venv/bin/python probe.py`. Il affiche salle, PV, position, mobs tués et la grille vue par l'IA (`#` mur, `~` eau ou trou, `.` libre, `P` PNJ, `E` ennemi).

## Go-Explore

L'entraînement garde une archive (`runs/<run>/archive/`) avec un save state par salle découverte, la clé étant (groupe, salle, saison). Elle démarre avec `states/*.state`. À chaque génération :
- 3 salles de départ sont tirées dans l'archive (`--starts`), en favorisant celles qui ont été le moins choisies. Tous les génomes jouent depuis ces mêmes salles, et la fitness est la moyenne.
- Quand un génome entre dans une salle absente de l'archive, on sauve un state dès que Link y est contrôlable. La salle rejoint l'archive et le génome est copié dans `hall/`, avec la salle dans le nom du fichier.
- Une salle absente de l'archive rapporte 200 points de plus qu'une salle déjà connue.
- Après 50 générations sans salle nouvelle (`--plateau`), on repart d'une nouvelle population.

L'archive est sur disque, elle survit aux reprises.

## Entraîner

    .venv/bin/python train.py                 # nouveau run dans runs/horon
    .venv/bin/python train.py --resume        # reprend au dernier checkpoint
    .venv/bin/python train.py --resume runs/horon/ckpt-40
    .venv/bin/python train.py --run runs/essai2

Ctrl+C sauve un checkpoint avant de quitter.

Dans `runs/<nom>/` :
- `ckpt-N` : population complète, toutes les 5 générations et toutes les 15 min, les 5 derniers sont gardés.
- `best.pkl` : dernier génome à avoir découvert une salle, `best.json` le nombre de salles de l'archive.
- `hall/` : chaque génome découvreur, jamais effacé.
- `kills.json` : kills cumulés (NEAT et aléatoire) et types d'ennemis tués. Le log les affiche aussi à chaque génération.
- `archive/` : un state par salle découverte, plus `index.json`.

## Choisir le nombre de workers

    .venv/bin/python bench_workers.py
    .venv/bin/python bench_workers.py --workers 4,6,8,10

Chaque génome joue un budget fixe de frames sur le premier state. Le script affiche génomes/s, frames/s et le gain par rapport à 1 worker. Choisis le plus petit nombre après lequel frames/s ne monte plus, puis passe-le à `train.py --workers N`. À refaire sur la vraie ROM, car la ROM factice est bien plus légère à émuler.

## Regarder

    .venv/bin/python replay.py
    .venv/bin/python replay.py --genome runs/horon/hall/gen00042-fit310.pkl

## Fitness

`zneat/fitness.py` : 200 par salle (groupe, salle) absente de l'archive, 100 par salle visitée (la saison ne compte pas), 10 par salle revue dans une autre saison, 100 par type d'ennemi tué, 100 par bit de progression (trésors, essences, flags globaux), -20 à la mort. Le but est le nombre de salles : les cases ne rapportent rien, elles servent seulement à la patience. Pas de bonus de survie : la mort arrête déjà la partie. Une partie dure 2 min de jeu max et s'arrête après 15 s sans nouvelle case, salle, ennemi ou flag (`zneat/runner.py`).

Adresses RAM dans `zneat/ram.py`, tirées de oracles-disasm.

## Joueur seul

Le score solo mesure ce qu'un réseau sait faire sans aide : il part de `states/horon-endgame.state`, joue 10 min de jeu (`SOLO_SECONDS`) sans save state ni arrêt pour patience, seulement la mort. Score = salles (groupe, salle) visitées, sans compter les saisons, + types d'ennemis tués. Le log affiche aussi les salles x saison et les bits d'histoire.

- `make start` : toutes les 5 générations (`SOLO_EVERY`), le génome au meilleur fitness de la génération joue sa partie solo dans le pool, en tâche de fond. La ligne `solo :` arrive dans le log quelques générations plus tard. `make status` affiche la dernière.
- `make solo GENOME=runs/horon/hall/xxx.pkl` : n'importe quel génome.

## Chemins d'inputs

Chaque cellule ajoutée à l'archive garde sa cellule parente (celle d'où la partie est partie) et les inputs joués depuis le state parent, dans `archive/<clé>.path.z` (un octet par pas de 12 frames, bits haut, bas, gauche, droite, A, B). En remontant les parents jusqu'à la graine, on obtient la suite d'inputs qui mène de Horon à la cellule. Le rejeu est exact (vérifié au pixel près par `make paths-check`), parce que `env.load` relâche toutes les touches après un chargement.

Les cellules d'avant cette version n'ont pas de chemin : `runs/horon` n'en a donc presque aucun. Pour le PPO, il faut un run qui a démarré avec les chemins, par exemple :

    make start RUN=runs/horon2
    make paths RUN=runs/horon2

## PPO : un seul réseau

`ppo.py` entraîne un seul réseau par PPO : un CNN sur l'écran (72x80 en niveaux de gris), un MLP sur un vecteur, puis un LSTM pour la mémoire. Le vecteur est l'observation de NEAT (grille de tuiles 9x9 et extras de `zneat/env.py`) plus de quoi savoir où il est (`Observer` dans `zneat/ppo_env.py`) : groupe, salle (coordonnées sur la grille 16x16 et 8 bits), position de Link dans la salle, et pour les 4 salles voisines si elles ont déjà été vues pendant l'épisode. NEAT garde ses 91 entrées, pour ne pas casser les runs existants. 15 actions (rien, 4 directions, A, B, direction + A, direction + B). Les émulateurs tournent dans `PPO_WORKERS` processus CPU (`ENVS` émulateurs chacun), le réseau sur le GPU s'il y en a un (venv ROCm `/mnt/data/dev/trainvenv-rocm`), sinon sur CPU.

Départs (phase 2 de Go-Explore, "backward algorithm") :
- Les `DEMOS` cellules de l'archive les plus avancées et joignables depuis Horon servent de cibles. Leur chemin est rejoué une fois et on sauve un state tous les 32 pas dans `PPO_RUN/demos/<clé>/`.
- Chaque démo commence près de sa fin. Un épisode réussit s'il gagne au moins autant que la démo depuis ce point (salles, types d'ennemis, histoire). Quand 25 % des 16 derniers épisodes réussissent, le départ recule de 10 s vers Horon. Une démo résolue depuis Horon est remplacée par une nouvelle cible (l'archive est relue).
- 20 % des épisodes partent de Horon (`HORON`).
- Sans aucun chemin dans l'archive, les départs sont tirés au hasard dans ses cellules, sans recul.

Récompense par épisode : +1 par salle nouvelle, +1 par type d'ennemi tué pour la première fois, +1 par bit d'histoire, +0,1 par salle dans une saison nouvelle (sinon changer de saison sur place rapporterait), -1 à la mort. Les cases ne rapportent rien. Un épisode s'arrête à la mort, après 10 min, ou après 30 s sans rien de nouveau. Pour un départ de démo, les salles et ennemis déjà vus par la démo avant ce point ne rapportent rien.

Imitation : chaque démo garde aussi l'écran, le vecteur et l'action à chaque pas (`frames.npy`, `vec.npy`, `act.npy`). À chaque nouvelle démo, le réseau fait 300 pas d'imitation pure (`BC_BURST`), puis une perte d'imitation (`BC_COEF`) s'ajoute au PPO à chaque époque. Sur ROCm, les convolutions passent par im2col + matmul (`Conv` dans `zneat/ppo_model.py`), 4 fois plus rapide que MIOpen.

Dans `PPO_RUN/` : `model.pt` (le réseau), `ckpt.pt` (optimiseur et curriculum, toutes les 5 min et à l'arrêt), `model-best.pt` et `solo-best.json` (meilleur score solo), `train.log`. Toutes les 15 min (`SOLO_MINUTES`), un processus à part fait jouer `model.pt` seul depuis Horon, échantillonné avec une graine fixe puis glouton, et écrit la ligne `solo :`.

    make ppo-start ARCHIVE=runs/horon2/archive
    make ppo-start PPO_WORKERS=22 ENVS=4      # serveur 24 cœurs

## Exploration sans NEAT

`make start` lance `explore.py` : Go-Explore par parties aléatoires seulement. À chaque tour, 50 cellules par worker sont tirées dans l'archive et une partie aux actions aléatoires collantes part de chacune. C'est bien plus rapide que NEAT pour trouver des salles (1 500 cellules en 90 s sur 8 cœurs). NEAT reste disponible avec `make neat`.

## NEAT en score solo

`make neat-spawns-start` lance `train.py --spawns spawns` dans `runs/neat-spawns`. À chaque génération, `NEAT_STARTS` cellules (5) sont tirées au hasard dans `spawns/`, et tous les génomes jouent seuls depuis ces mêmes départs, 2 min au plus, arrêt après 15 s sans rien de nouveau. La fitness ne compte que ce que le génome a fait lui-même : 100 par salle, 100 par type d'ennemi tué, 100 par flag d'histoire, 10 par salle déjà vue dans une autre saison, -20 s'il meurt. Il n'y a ni archive commune, ni bonus de nouveauté, ni parties aléatoires.

Toutes les 5 générations, le meilleur génome joue 10 min seul depuis Horon (ligne `solo :`). Le meilleur à ce test est gardé dans `runs/neat-spawns/best-solo.pkl` (score dans `solo-best.json`). Pour le comparer au PPO :

    make compare GENOME=runs/neat-spawns/best-solo.pkl
    make compare GENOME=runs/neat-spawns/best-solo.pkl RANDOM_LOCATION=1

## Dataset world model

Le dépôt contient ce qu'il faut pour générer le dataset sans rien réentraîner :
- `models/ppo.pt` : le modèle officiel, seul modèle du dépôt. Joueur PPO (CNN Impala sur l'écran en moyenne 2x2 + grille de la salle + LSTM), 12 frames par action, 20 M pas (run `iter2`). Sur 30 parties de 2 min téléportées au hasard : 5,3 salles (max 24), 0,9 kill, 17 % de morts, 1,4 % du temps dans le menu. Détails dans `models/ppo.json`.
- `spawns/` : 1 054 points de départ (states PyBoy compressés), au plus 8 par salle, sur 134 salles, tirés de l'archive Go-Explore `runs/horon/archive`.
- `states/horon-endgame.state` : Horon avec l'inventaire de fin de jeu.

La ROM n'est pas fournie. Il faut la version européenne d'Oracle of Seasons dans `rom/oos.gbc`.

Installation, puis lancement :

    ./install.sh                     # .venv avec PyBoy
    make install-ppo                 # torch CPU dans .venv (inutile si le venv ROCm existe)
    make dataset-start               # en tâche de fond, log dans DATASET/dataset.log
    make status                      # nombre de parties et taille
    make dataset-stop                # arrêt (la partie en cours est perdue, les autres restent)

`make dataset` fait la même chose au premier plan. Les réglages se passent en variables :

    make dataset-start DATASET=/chemin/zelda DATASET_GB=50 DATASET_WORKERS=10

| Variable | Défaut | Rôle |
|---|---|---|
| `DATASET` | `/mnt/data/datasets/zelda` | dossier de sortie |
| `DATASET_GB` | 50 | arrêt quand le dossier atteint cette taille |
| `DATASET_WORKERS` | cœurs libres | un émulateur par worker ; par défaut nproc moins la charge du moment |
| `DATASET_MODEL` | `models/ppo.pt` | joueurs, à parts égales s'il y en a plusieurs ; vide = tout au hasard |
| `DATASET_MODEL_FRAC` | 0.7 | part des parties jouées par le modèle, le reste au hasard |
| `DATASET_ARCHIVE` | `spawns` | cellules de départ |

Chaque partie dure 2 min de jeu (`--seconds` de `dataset.py`) et part d'une cellule de `spawns/` tirée au hasard, ou de Horon une fois sur 10 (`--horon`). Elle est jouée par un modèle 7 fois sur 10 (`models/ppo.pt`, échantillonné, graine différente à chaque partie), et sinon par des actions aléatoires collantes : le modèle donne des trajectoires qui vont quelque part, le hasard couvre les actions que le modèle ne prend jamais (coups dans le vide, marche contre les murs). La partie s'arrête plus tôt si Link meurt. Une config `config-<graine>.json` est écrite à chaque lancement, on peut en lancer plusieurs sur le même dossier.

Un fichier par partie, `ep-<graine>-<worker>-<n>.npz` :
- `frames` (N, 144, 160, 3) uint8 RGB, l'écran après chaque action ;
- `actions` (N,) uint8, bits haut, bas, gauche, droite, A, B, tenus 12 frames (`frames_per_action`, absent des parties enregistrées avant ce changement, qui sont à 4 frames) ;
- `scripted` (N,) bool, pas joués par la séquence de menu scriptée (`--menu-frac`, 5 % par défaut) plutôt que par la politique ;
- `first` l'écran avant la première action ;
- `start` la cellule de départ, `policy` (`ppo` ou `random`).

Environ 3,4 Ko par frame compressée : 50 Go font à peu près 14 millions de frames, soit 270 h de jeu à 15 actions par seconde.

Pour repartir d'une archive plus grosse (après un `make start` qui a tourné longtemps) : `make dataset-start DATASET_ARCHIVE=runs/horon/archive`, `dataset.py` y prend aussi 8 cellules par salle au plus.
