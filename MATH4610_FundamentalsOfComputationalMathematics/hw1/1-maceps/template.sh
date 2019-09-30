#!/usr/bin/env bash
echo "clang++ -Xclang -ast-print -fsyntax-only -std=c++17 main.cpp > template.txt"
clang++ -Xclang -ast-print -fsyntax-only -std=c++17 main.cpp > template.txt
