# infer-day-tune-night

Plateforme Kubernetes générique : un SLM **sert les requêtes le jour** et
**s'auto-affine la nuit** (LoRA) sur les échanges de la journée, sans intervention
humaine. Un adapter n'est promu que s'il passe la porte d'évaluation.

- **KEDA** : inférence chaude le jour, scale-to-zero et réveil à la demande la nuit.
- **KAI Scheduler** : queues `idtn-inference` (prioritaire) / `idtn-training`
  (préemptible), arbitrage de l'accélérateur.
- **Profils matériels** : `nvidia`, `amd-rocm`, `apu-vulkan`, `cpu`.

```
 JOUR                                          NUIT
 client ─▶ gateway ─▶ inference (vLLM | llama.cpp + LoRA courant)
              │ JSONL                         night-pipeline (CronJob, queue KAI training)
              └──────▶ /data/logs ─▶ curate ─▶ train ─▶ eval ─▶ promote ─▶ adapters/current
```

## Arborescence

```
config/
  platform.yaml               configuration commune (horaires, LoRA, porte d'éval, garde-fous)
  profiles/<profil>.yaml      surcharges matérielles (modèle, backend, device)
cluster/
  addons/install.sh           KEDA + KAI + device plugin du profil
  addons/kai-queues.yaml      hiérarchie de queues KAI
  addons/device-plugins/      plugin générique /dev/dri, /dev/accel (APU, NPU)
  local-k3s/                  cluster local de dev/démo
deploy/
  base/                       gateway, ScaledObject KEDA, CronJob de nuit, bootstrap, PVC, RBAC
  components/inference-*/     vLLM ou llama.cpp (interchangeables)
  overlays/<profil>/          assemblage + ressources accélérateur
images/                       gateway, trainer (CPU / CUDA / ROCm via build-args)
src/idtn/
  conductor.py                CLI des Jobs : bootstrap | night | guards
  guards.py                   secteur, températures, RAM
  gateway/app.py              proxy OpenAI, journalisation, métrique KEDA
  pipeline/                   curate, judge, train_lora, evaluate, promote
docs/                         architecture, profils matériels
```

## Déploiement chez un client

```bash
make addons  PROFILE=nvidia KAI_VERSION=<version>
make images  TRAINER_ARGS="--build-arg BASE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --build-arg EXTRAS=train,cuda"
make deploy  PROFILE=nvidia
kubectl -n idtn create secret generic hf-token --from-literal=token=...   # si modèle restreint
make night-now                                                             # test d'un cycle
```

Le client fournit idéalement un **golden set** (`/data/golden/golden.jsonl`,
au format `{"messages": [...]}`) : c'est le garde-fou contre la dérive du modèle.

Voir [docs/architecture.md](docs/architecture.md) et [docs/hardware-profiles.md](docs/hardware-profiles.md).
