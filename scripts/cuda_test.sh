#!/usr/bin/env bash

set -xe

BRANCH_NAME=cron-jobs

# On exit, go back to `dev` branch and remove `cron-jobs` branch
function _cleanup() {
  [[ ${#DIRSTACK[@]} -gt 1 ]] && popd
  git rev-parse --verify "${BRANCH_NAME}"
  git checkout dev
  if [[ $? == 0 ]]; then
    git branch -D "${BRANCH_NAME}"
  fi
}

trap _cleanup EXIT

VIRTUALENV=pyro-cron-27
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd "${DIR}"
source activate "${VIRTUALENV}"
git checkout dev
git pull upstream dev
git fetch origin "${BRANCH_NAME}"
git checkout "${BRANCH_NAME}"
git rebase dev

make test-cuda
