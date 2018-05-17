#!/usr/bin/env bash

set -xe

REF_HEAD_FILE=.cron/ref_head.txt
BRANCH_NAME=cron-jobs
VIRTUALENV=pyro-cron-27

ref=$(<${REF_HEAD_FILE})
source activate "${VIRTUALENV}"
git checkout dev
git pull upstream dev
git checkout "${BRANCH_NAME}"
git pull origin "${BRANCH_NAME}"
git rebase dev
bash perf_test.sh "${ref}"
if [[ $? -ne 0 ]]; then
  cur_head=$(git rev-parse --abbrev-ref HEAD)
  echo "${cur_head}" >| "${REF_HEAD_FILE}"
fi
