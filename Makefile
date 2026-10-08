PROFILE ?= cpu
TAG     ?= dev
KAI_VERSION ?=
KUSTOMIZE = kubectl kustomize --load-restrictor LoadRestrictionsNone

.PHONY: help local-cluster addons images deploy render bootstrap night-now status test lint

help:            ## Affiche cette aide
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

local-cluster:   ## k3s local mono-nœud (dev/démo uniquement)
	./cluster/local-k3s/install.sh

addons:          ## KEDA + KAI + device plugin du profil (PROFILE=, KAI_VERSION=)
	PROFILE=$(PROFILE) KAI_VERSION=$(KAI_VERSION) ./cluster/addons/install.sh

images:          ## Construit les images (TRAINER_ARGS pour CUDA/ROCm)
	docker build -t idtn/gateway:$(TAG) -f images/gateway/Dockerfile .
	docker build -t idtn/trainer:$(TAG) -f images/trainer/Dockerfile $(TRAINER_ARGS) .

render:          ## Affiche les manifests du profil
	$(KUSTOMIZE) deploy/overlays/$(PROFILE)

deploy:          ## Déploie le profil
	$(KUSTOMIZE) deploy/overlays/$(PROFILE) | kubectl apply -f -

night-now:       ## Lance un cycle de nuit immédiatement
	kubectl -n idtn create job --from=cronjob/night-pipeline night-manual-$$(date +%s)

status:          ## État de la plateforme
	kubectl -n idtn get scaledobject,deploy,cronjob,job,pod

test:            ## Tests unitaires
	uv run --extra dev pytest -q

lint:
	uv run --extra dev ruff check src tests
