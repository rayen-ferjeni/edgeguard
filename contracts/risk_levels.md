# Niveaux de risque EdgeGuard

| Niveau  | Signification | Action |
|---------|---------------|--------|
| NORMAL  | Aucun signal anormal | Aucune. Affichage vert. |
| WARNING | Signal faible, ou une source ne répond plus (mode dégradé) | Alerte affichée et journalisée. La machine continue. |
| DANGER  | Situation dangereuse confirmée (ex. personne en zone + machine en marche, anomalie forte, fumée confirmée) | Alarme + demande d'arrêt. Le STM32 coupe le moteur. |

## Règles
- Le niveau est une information produite par le logiciel (Jetson / PC).
- Seul le STM32 arrête réellement le moteur. Si la Jetson ou le PC tombe, le STM32 passe seul en état sûr (perte du heartbeat).
- Chaque alerte indique quelles règles ou quels capteurs l'ont déclenchée (décision explicable).
- Un LLM ou un agent ne décide jamais du niveau de risque et ne commande jamais le matériel.
