#!/usr/bin/env bash
# k3s local dédié (dev/démo). N'interfère pas avec d'autres contextes kubectl.
set -euo pipefail
sudo install -D -m 0644 "$(dirname "$0")/config.yaml" /etc/rancher/k3s/config.yaml
if command -v k3s >/dev/null; then
  sudo systemctl enable --now k3s
else
  curl -sfL https://get.k3s.io | sh -
fi
mkdir -p ~/.kube
sudo k3s kubectl config view --raw | sed 's/name: default/name: idtn-local/; s/cluster: default/cluster: idtn-local/; s/user: default/user: idtn-local/; s/current-context: default/current-context: idtn-local/' > ~/.kube/idtn-local.yaml
echo "export KUBECONFIG=~/.kube/idtn-local.yaml"
