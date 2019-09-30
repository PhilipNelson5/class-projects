#!/usr/bin/env bash
Black='\033[0;30m'        # Black
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple
Cyan='\033[0;36m'         # Cyan
White='\033[0;37m'        # White
RESET='\033[0m'

if [ $# -eq 0 ]; then
  echo "Error: no file specified"
  exit 1
fi

kwargs="${@:2}"

file=$(find test -type f | grep --invert-match '/lexical/' | grep $1 | head -n 1)

echo -e "$Cyan[ COMPILING ]$RESET cpsl < ${file}"

build/cpsl $kwargs < ${file}

echo ""

echo $file
printf '━%.0s' $(seq 1 $(tput cols))
echo ""

cat -n $file

printf '━%.0s' $(seq 1 $(tput cols))
echo ""
echo ""
