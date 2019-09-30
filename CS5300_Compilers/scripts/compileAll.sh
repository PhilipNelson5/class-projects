#!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
RESET='\033[0m'
if [ $# -eq 0 ]; then
  echo "Error: no directory specified"
  exit 1
fi

kwargs="${@:2}"

dir=$(find test -type d | grep $1 | head -n 1)

if [[ ! -d "${dir}" ]]; then
    echo -e "$Red[ Error ]$RESET $1 did not match any directory";
fi

clear
for file in $(ls ${dir});
do
  path="${dir}/${file}"
  echo -e "$Cyan[ COMPILING ]$RESET cpsl < ${path}"
  build/cpsl $kwargs < ${path} > output.asm

  printf '─%.0s' $(seq 1 $(tput cols))
  echo -e "$Cyan[ MARS ]$RESET java -jar ~/Downloads/Mars4_5.jar output.asm"
  printf '─%.0s' $(seq 1 $(tput cols))

  java -jar ~/Downloads/Mars4_5.jar output.asm

  printf '━%.0s' $(seq 1 $(tput cols))

  printf '─%.0s' $(seq 1 $(tput cols))
  echo "File:  ${path}"
  printf '─%.0s' $(seq 1 $(tput cols))
  echo ""

  cat -n $path

  printf '━%.0s' $(seq 1 $(tput cols))
  echo ""
  echo ""
  read -p "Press enter to continue"
  clear
done

exit 0
