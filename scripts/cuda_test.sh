#!/usr/bin/env bash

set -xe

BRANCH_NAME=cron-jobs
VIRTUALENV=pyro-cron-27

source activate "${VIRTUALENV}"
git checkout dev
git pull upstream dev
git checkout "${BRANCH_NAME}"
git pull origin "${BRANCH_NAME}"
git rebase dev
make test-cuda
