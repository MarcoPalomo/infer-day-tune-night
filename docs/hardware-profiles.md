# Profils matériels

Chaque client choisit **un profil**, qui détermine :

- l'overlay kustomize (`deploy/overlays/<profil>`) ;
- le fichier de surcharge (`config/profiles/<profil>.yaml`) ;
- le device plugin installé par `cluster/addons/install.sh`.

| Profil | Accélérateur | Device plugin | Inférence | Entraînement nocturne | SLM par défaut |
|---|---|---|---|---|---|
| `nvidia` | GPU NVIDIA | NVIDIA GPU Operator | vLLM | QLoRA CUDA | 7–8 B |
| `amd-rocm` | Instinct / Radeon pris en charge par ROCm | AMD GPU Operator | vLLM (ROCm) | LoRA ROCm | 7–8 B |
| `apu-vulkan` | iGPU / APU (AMD Radeon 7xxM/8xxM, Strix Halo, Intel Arc iGPU) | generic-device-plugin (`/dev/dri`) | llama.cpp Vulkan | LoRA CPU | 0,5–1,5 B |
| `cpu` | aucun | – | llama.cpp CPU | LoRA CPU | 0,5 B |

## Notes par profil

### nvidia
C'est le profil où KAI apporte le plus : partage fractionnaire de GPU, queues
par équipe, gang scheduling pour l'entraînement multi-GPU.

### amd-rocm
KAI gère bien les queues et les priorités. En revanche, `gpu-fraction` cible
NVIDIA : il faut le vérifier par version de KAI. Il faut aussi contrôler que
le GPU visé figure dans la matrice de support ROCm.

### apu-vulkan
La mémoire est partagée avec l'hôte, le modèle reste donc petit.
L'entraînement ROCm sur iGPU n'est pas officiel : un hang GPU peut figer
toute la machine. C'est pourquoi l'entraînement se fait sur CPU par défaut.
Les garde-fous secteur et température sont activés.

### NPU (AMD XDNA, Intel NPU) : extension future
Aujourd'hui, le support LLM des NPU sous Linux se limite à l'inférence de
modèles pré-quantifiés (ONNX Runtime + VitisAI EP, OpenVINO, FastFlowLM).
Il ne permet **pas d'entraînement**. La piste prévue est un composant
`inference-npu` qui servirait le jour sur le NPU, pour libérer le GPU ou
l'iGPU. Le device plugin générique peut déjà exposer `/dev/accel/*`.

## Machine de référence de développement

Lenovo 83KG : Ryzen AI 7 350, Radeon 860M (gfx1152), NPU XDNA 2, 30 Gio de RAM.
Elle correspond au profil `apu-vulkan` et tourne sur un k3s local
(`cluster/local-k3s`).
