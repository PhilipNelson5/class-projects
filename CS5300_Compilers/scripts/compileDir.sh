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
  echo -e "$Red[ Error ]$RESET no directory specified"
  exit 1
fi

kwargs="${@:2}"

dir=$(find test -type d | grep $1 | head -n 1)

if [[ ! -d "${dir}" ]]; then
    echo -e "$Red[ Error ]$RESET $1 did not match any directory";
fi

echo "$dir"

clear
for file in $(ls ${dir});
do
  path="${dir}/${file}"
  echo -e "$Cyan[ COMPILING ]$RESET cpsl < ${path}"
  build/cpsl $kwargs < ${path}

  echo ""

  echo $path
  printf '━%.0s' $(seq 1 $(tput cols))
  echo ""

  cat -n $path

  printf '━%.0s' $(seq 1 $(tput cols))
  echo ""
  echo ""
  read -p "Press enter to continue"
  clear
done

exit 0

