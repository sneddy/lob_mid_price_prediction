#!/usr/bin/env bash

set -e

# Names to identify images and containers of this app
TAG_NAME=xtx-${USER}
CONTAINER_NAME=xtx_${USER}
PROJECT_NAME=/xtx

# Output colors
NORMAL="\\033[0;39m"
RED="\\033[1;31m"
BLUE="\\033[1;34m"

# logging base configuration
LOG_LEVEL="DEBUG"

log() {
  echo -e "$BLUE > $1 $NORMAL"
}
build() {
    echo "disabled buildkit - for base images pulling..."
    DOCKER_BUILDKIT=0 docker build \
      --pull \
      --build-arg USER=${USER} \
      --build-arg UID=`id -u` \
      --build-arg GID=`id -g` \
      -t "${TAG_NAME}" . || true

    echo "enabled buildkit - for installing dependencies from private repos"
    DOCKER_BUILDKIT=1 docker build \
      --build-arg USER=${USER} \
      --build-arg UID=`id -u` \
      --build-arg GID=`id -g` \
      -t "${TAG_NAME}" .
    [ $? != 0 ] && error "Docker image build failed !" && exit 100
}

run() {
    docker run \
        -e DISPLAY=unix${DISPLAY} \
        -e PYTHONWARNINGS="ignore:Unverified HTTPS request,ignore:resource_tracker:UserWarning" \
        -e LOG_LEVEL=${LOG_LEVEL} \
        -e HOSTNAME="$(hostname)" \
        -e USERNAME="${USER}" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        --privileged \
        --ipc=host \
        -itd \
        --name="${CONTAINER_NAME}" \
        -v /etc/xdg/dvc:/etc/xdg/dvc:ro \
        -v "${PWD}":${PROJECT_NAME} \
        "${TAG_NAME}" bash
}

exec() {
  docker exec -it ${CONTAINER_NAME} bash
}

stop() {
    log "Stopping and removing the container ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}"; docker rm "${CONTAINER_NAME}"
}

lint() {
    log "Linting"
    pre-commit run --all-files
}

$*
