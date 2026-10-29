# Plan zelda-neat

Pour reprendre dans une autre conversation : lire ce fichier, puis README.md et le Makefile.

## But

Un seul modèle joueur qui part de Horon (`states/horon-endgame.state`) et va le plus loin possible tout seul, puis un dataset (frames + inputs) pour entraîner un world model.

Ce qui compte : des salles différentes ("maps"), pas des cases. Score solo = salles (groupe, salle) visitées en 10 min depuis Horon + types d'ennemis tués. La saison compte très peu (sinon changer de saison sur place rapporte).

## État au 2026-10-29 soir

Fait et testé :
- Métrique solo : `make solo` (génome NEAT), `make ppo-solo` (modèle PPO), ligne `solo :` dans les logs et dans `make status`.
- Chemins d'inputs dans l'archive : chaque cellule garde sa parente et les inputs depuis le state parent (`archive/<clé>.path.z`). Rejeu exact au pixel près (`make paths-check`).
- Exploration Go-Explore par parties aléatoires seulement (`explore.py`, `make start`). NEAT abandonné : son meilleur génome faisait 2 salles seul. Toujours disponible avec `make neat`.
- Joueur PPO (`ppo.py`) : CNN sur l'écran 72x80 gris + vecteur (grille 9x9, repères de salle, salles voisines déjà vues) + LSTM. Départs pris près de la fin des chemins de l'archive puis reculés vers Horon (backward algorithm), 20 % depuis Horon. Imitation des chemins : 300 pas à chaque nouvelle démo, puis perte d'imitation (poids 0,2) une fois par époque. Relit l'archive toutes les 10 min.
- Dataset : `dataset.py`, `make dataset-start`, dans `/mnt/data/datasets/zelda`, arrêt à 50 Go. Parties de 2 min depuis des cellules au hasard de l'archive, 70 % jouées par `runs/ppo/model-best.pt` s'il existe, le reste au hasard.

En cours sur la machine locale (8 cœurs, RX 9070 XT) :
- `zelda-explore` : `make start WORKERS=3`, log `runs/horon/train.log`.
- `zelda-ppo` : `make ppo-start PPO_WORKERS=4`, log `runs/ppo/train.log`, ~1 300 pas/s.
- Ancien run NEAT sans chemins gardé dans `runs/horon-sans-chemins` (à effacer si inutile).

Pas lancé : le dataset (les cœurs sont pris).

## Prochaines étapes

1. Suivre `make status` : le score solo PPO doit monter (ligne `solo :` toutes les 15 min, meilleur dans `runs/ppo/solo-best.json`). Référence : NEAT faisait 2.
2. Si le curriculum ne recule pas (pointeurs des démos figés dans les lignes `curriculum :`), baisser `--success` ou `--back-seconds`, ou monter `HORON`.
3. L'imitation apprend peu (perte ~2,3, près du hasard) parce que les actions de l'archive sont aléatoires. Idée : imiter plutôt des chemins raccourcis, ou baisser `BC_COEF`.
4. Quand `runs/ppo/model-best.pt` dépasse nettement le hasard : `make dataset-start` (ou sur le serveur 24 cœurs).
5. Option : renvoyer dans l'archive les salles nouvelles trouvées par le PPO, pour que l'explorateur reparte de plus loin.
6. Option si la vitesse limite : réduire le coût du CNN (écran plus petit, moins d'époques), ou passer l'entraînement sur une machine avec plus de cœurs pour les émulateurs.

## Pièges connus

- Convolutions MIOpen 4 fois plus lentes qu'im2col + matmul sur la RX 9070 (ROCm 6.4) : classe `Conv` dans `zneat/ppo_model.py`.
- PyBoy n'a pas de son émulé sans fenêtre : patch mémoire bank $39 dans `zneat/env.py`.
- `env.load` relâche toutes les touches, c'est ce qui rend les chemins rejouables à l'identique.
- Le venv ROCm (`/mnt/data/dev/trainvenv-rocm`, Python 3.14) a pyboy et neat-python installés. Sur le serveur sans GPU : `make install-ppo` (torch CPU dans .venv).
- Changer l'observation ou le réseau rend `runs/ppo` incompatible : `make ppo-reset CONFIRM=oui`.
- Les jobs lourds (agy, gros dumps) sous `systemd-run --user --scope -p MemoryMax=...`, sinon l'OOM peut tuer la session.
