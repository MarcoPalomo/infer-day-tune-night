#!/usr/bin/env bash
# Installe les add-ons requis sur un cluster existant (kubectl courant).
# Usage : PROFILE=nvidia|amd-rocm|apu-vulkan|cpu ./cluster/addons/install.sh
set -euo pipefail
cd "$(dirname "$0")"

PROFILE="${PROFILE:?PROFILE requis (nvidia|amd-rocm|apu-vulkan|cpu)}"
KEDA_VERSION="${KEDA_VERSION:-}"   # vide = dernière version du chart
KAI_VERSION="${KAI_VERSION:?KAI_VERSION requis, ex. v0.9.x (voir github.com/NVIDIA/KAI-Scheduler/releases)}"

ver() { [[ -n "$1" ]] && echo "--version $1"; }

echo "==> KEDA"
helm repo add kedacore https://kedacore.github.io/charts >/dev/null
helm repo update kedacore >/dev/null
helm upgrade --install keda kedacore/keda -n keda --create-namespace $(ver "$KEDA_VERSION") --wait

echo "==> KAI Scheduler"
helm upgrade --install kai-scheduler oci://ghcr.io/nvidia/kai-scheduler/kai-scheduler \
  -n kai-scheduler --create-namespace --version "$KAI_VERSION" --wait
kubectl apply -f kai-queues.yaml

echo "==> Device plugin ($PROFILE)"
case "$PROFILE" in
  nvidia)
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null
    helm repo update nvidia >/dev/null
    helm upgrade --install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace --wait
    ;;
  amd-rocm)
    # Pour la prod, préférer l'AMD GPU Operator (rocm.github.io/gpu-operator).
    kubectl apply -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml
    ;;
  apu-vulkan)
    kubectl apply -f device-plugins/generic-dri.yaml
    ;;
  cpu) echo "aucun" ;;
  *) echo "profil inconnu : $PROFILE" >&2; exit 1 ;;
esac

echo "==> PriorityClasses disponibles (attendu : inference, train)"
kubectl get priorityclass
