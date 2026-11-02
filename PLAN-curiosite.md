# Plan : explorateur curieux guidé par le world model (façon Plan2Explore)

Pour reprendre dans une autre conversation : lire ce fichier, puis PLAN.md, README.md et `runs/journal.md`.

## Idée

Boucle à deux joueurs, sans gradient entre eux :
- le world model apprend à prédire l'écran suivant à partir de l'écran et de l'action ;
- l'explorateur PPO reçoit une récompense là où plusieurs world models ne sont pas d'accord entre eux.

On récompense le désaccord d'un ensemble de modèles, pas l'erreur d'un seul. Sinon l'explorateur se fixe sur
ce qui est imprévisible par nature (« noisy TV » : ennemi au mouvement aléatoire, hasard du jeu), où l'erreur ne
baisse jamais. Sur du pur hasard, les modèles finissent par prédire la même moyenne et le désaccord retombe.
Sur ce qu'ils n'ont pas encore vu, ils divergent.

On fait des cycles, pas un entraînement simultané :
1. générer un dataset avec l'explorateur du moment ;
2. entraîner le world model (l'ensemble) sur tout ce qu'on a ;
3. réentraîner l'explorateur avec le désaccord en récompense en plus ;
4. recommencer.

On garde une part de parties normales dans le dataset (`models/ppo.pt` et hasard), sinon le world model ne voit
que des cas difficiles et oublie le jeu ordinaire.

## Première étape : poser les bases sur un micro dataset

Le but n'est pas encore un bon world model. Il faut que toute la boucle tourne de bout en bout, vite, et qu'on
sache mesurer si elle aide.

### 1. Micro dataset de référence

- `make dataset` avec `models/ppo.pt` (70 %) et le hasard (30 %), comme aujourd'hui, dans un dossier à part, par
  exemple `/mnt/data/datasets/zelda-micro/c0` (cycle 0), arrêté vers 300 parties de 2 min (environ 180 000 pas).
  Regarder la taille obtenue et ajuster.
- Mettre de côté un jeu de test fixe, jamais utilisé à l'entraînement : 30 parties, graine fixe, même mélange.
  C'est la mesure de vérité de tout le plan.

### 2. World model minimal (nouveau `wm/`)

Volontairement petit pour tourner en minutes sur la RX 9070 XT :
- entrée en 72x80 (la vue du modèle, `env.frame()`, moyenne 2x2), pas l'image native, pour les bases ;
- encodeur convolutif vers un vecteur z (256), avec un décodeur pour l'entraîner en autoencodeur
  (perte de reconstruction) ;
- ensemble de K = 5 têtes de dynamique (MLP) qui prédisent z(t+1) à partir de z(t), z(t-1) et de l'action
  (masque de 7 boutons du dataset, voir `to_mask`). Initialisations différentes, minibatchs différents ;
- désaccord = variance entre les 5 prédictions de z(t+1), moyennée sur les dimensions ;
- script `wm/train.py --data <dossiers> --out runs/wm/cN/wm.pt`, plus une éval sur le jeu de test :
  erreur de prédiction de z(t+1) et image reconstruite.

Attention aux convolutions MIOpen lentes : reprendre la classe `Conv` (im2col + matmul) de `zneat/ppo_model.py`.

### 3. Désaccord en récompense dans le PPO

- Option `--curiosity runs/wm/cN/wm.pt --beta 0.x` dans `ppo.py`. Le world model est chargé figé sur le GPU.
- Branchement dans la boucle de collecte (`ppo.py`, autour de `vec.step`) : on a déjà sur le GPU l'écran avant
  (`frame_buf[t]`), l'action `a` et l'écran après (`frames`). Les écrans du PPO sont en 72x80, comme le world
  model. Il faut garder z(t-1) par environnement.
- Récompense intrinsèque = désaccord, normalisé par son écart type courant (comme `RewardNorm`), à zéro quand
  l'épisode se termine (`done`), multiplié par beta, ajouté à la récompense actuelle.
- Le GPU attend environ 70 % du temps (la collecte domine), donc ce calcul doit être presque gratuit. À vérifier
  dans la ligne `collecte X %`.
- Partir des poids de `models/ppo.pt` (copier son ckpt comme pour `iter2`), quelques millions de pas suffisent
  pour voir l'effet.
- Logger la récompense intrinsèque moyenne par partie, à côté de salles, types et morts.

### 4. Boucler un cycle et mesurer

Cycle 1 : générer `c1` avec l'explorateur curieux (même taille que `c0`), puis entraîner deux world models à
budget égal :
- A sur `c0` + un autre lot normal de même taille ;
- B sur `c0` + `c1`.

Réussite si B prédit mieux que A sur le jeu de test fixe, alors que ce jeu ne contient que des parties normales.
C'est la seule mesure qui compte : le dataset curieux doit aider le world model, pas juste avoir l'air varié.

À suivre aussi :
- `scripts/eval_model.py` sur l'explorateur : salles, types, morts, menu, il ne doit pas s'effondrer ;
- ce qu'il fait vraiment : `make ppo-watch MODEL=...`. S'il se fige devant un ennemi ou une animation, c'est le
  noisy TV qui revient, baisser beta ou monter K ;
- le désaccord moyen sur le jeu de test, qui doit baisser de cycle en cycle.

### 5. Ce qu'on décide après les bases

- Si B bat A : passer à l'échelle (vrai dataset, image native pour le world model final), et faire plusieurs
  cycles.
- Beta et la part de parties normales sont de bons candidats pour le mode hybride (population de PPO à
  réglages qui évoluent).
- Plus tard, remplacer le world model minimal par le vrai world model du projet, s'il a une sortie qui permet un
  ensemble ou un désaccord.

## Pièges à garder en tête

- Le world model ne voit que 72x80 dans ces bases. Le dataset, lui, reste en natif 144x160 (`dataset.py`).
- Changer l'observation du PPO rend les runs incompatibles. Ici on n'y touche pas : la curiosité n'entre que dans
  la récompense.
- `dataset.py` enregistre `scripted` pour les pas joués par la séquence de menu : les garder pour le world model,
  mais ne pas les compter dans les stats de l'explorateur.
- Les 8 cœurs sont le goulot pour la collecte. Ne pas lancer dataset, PPO et NEAT en même temps.
