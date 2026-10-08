# Architecture

## Principe

Un SLM unique par déploiement :

- **Le jour**, il sert des requêtes via une API compatible OpenAI.
- **La nuit**, il s'affine (LoRA) sur les échanges de la journée.
- **Aucun humain dans la boucle** : un nouvel adapter n'est mis en production que
  s'il passe une porte d'évaluation automatique, et le rollback se fait par
  simple bascule de symlink.

```
 JOUR                                          NUIT
 client ─▶ gateway ─▶ inference (vLLM | llama.cpp + LoRA courant)
              │ JSONL                         night-pipeline (CronJob, queue KAI training)
              └──────▶ /data/logs ─▶ curate ─▶ train ─▶ eval ─▶ promote ─▶ adapters/current
```

## Briques

| Brique | Rôle | Implémentation |
|---|---|---|
| Gateway | API OpenAI, journalisation, maintien des requêtes pendant un réveil depuis 0 | `src/idtn/gateway` (FastAPI) |
| Inference | Service du modèle de base + adapter LoRA courant | composant `inference-vllm` ou `inference-llamacpp` |
| **KEDA** | Cycle de vie de l'inférence : cron le jour, scale-to-zero et réveil à la demande la nuit | `deploy/base/keda.yaml` |
| **KAI Scheduler** | Queues hiérarchiques, priorités, préemption training ↔ inference, gang scheduling | `cluster/addons/kai-queues.yaml` |
| Night pipeline | curate → train → eval → promote | CronJob + `src/idtn/pipeline` |
| Guards | Arrêt propre sur seuils (secteur, températures, RAM) | `src/idtn/guards.py` |

## KEDA et KAI : partage des rôles

```
KEDA  ── décide COMBIEN de pods d'inférence (0 ou 1, selon l'heure et la demande)
KAI   ── décide QUI obtient l'accélérateur quand ça se dispute
           queue idtn
            ├─ idtn-inference   priorityClass inference   non préemptible
            └─ idtn-training    priorityClass train       préemptible
```

Quand une requête arrive la nuit, la gateway la garde en attente et expose
`pending > 0` sur `/keda/metrics`. KEDA remonte alors l'inférence à 1 réplique.
Si le GPU est occupé par l'entraînement, KAI préempte le Job d'entraînement, qui
reprendra depuis son dernier checkpoint.

Sur les profils NVIDIA avec plusieurs GPU, ou avec du partage fractionnaire
(`gpu-fraction` de KAI), l'inférence et l'entraînement peuvent même cohabiter
sans préemption.

## Porte d'évaluation

- Perplexité sur le **holdout** du jour : il faut un gain d'au moins `eval.min_holdout_gain`.
- Perplexité sur le **golden set** figé, fourni par le client : la régression doit
  rester inférieure à `eval.max_golden_regression`.
- Si l'une des deux conditions échoue, l'adapter est archivé sans être promu.

## Données et conformité

Les logs de la gateway contiennent les données des clients finaux. La curation
retire les données personnelles (PII) avant tout entraînement. Durée de
rétention et chiffrement du volume sont à définir avec chaque client
(voir `paths.logs`).

Voir [hardware-profiles.md](hardware-profiles.md) pour le choix du profil.
