#!/usr/bin/env bash
RED='\033[0;31m'
GREEN='\033[0;32m'
RESET='\033[0m'
passed=0;
tests=0;
for file in $(find test -type f | grep --invert-match '/lexical/'); do
  output=$(build/cpsl < ${file} 2>&1)

  if [[ -z ${output} ]]; then
    echo -e "$GREEN[ PASS ]$RESET cpsl < ${file}"
    ((++passed))
  else
    #printf '━%.0s' $(seq 1 $(tput cols))
    #echo ""

    echo -e "$RED[ FAIL ]$RESET cpsl < ${file}"
    echo "${output}"

    echo ""
  fi

  ((++tests))
done

echo "Passed ${passed} / ${tests} tests"
