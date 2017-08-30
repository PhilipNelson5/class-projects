#!/bin/bash
# Updates the tree directories in the repo READMEs

GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "Root dir:"
echo ~~~~ > README.md
tree -d -L 2 --noreport >> README.md
cat README.md
echo " "

echo "CS1400_CS1"
cd CS1400_CS1/
echo ~~~~ > README.md
tree -d -L 2 --noreport >> README.md
cat README.md
echo " "

echo "CS1410_CS2"
cd ../CS1410_CS2/
echo ~~~~ > README.md
tree -d -L 2 --noreport >> README.md
cat README.md
echo " "

echo "CS2410_JAVA"
cd ../CS2410_JAVA/
echo ~~~~ > README.md
tree -d -L 2 --noreport >> README.md
cat README.md
echo " "

echo "CS2420_CS3"
cd ../CS2420_CS3/
echo ~~~~ > README.md
tree -d -L 2 --noreport >> README.md
cat README.md
echo " "

echo "CS3100_OperatingSystems"
cd ../CS3100_OperatingSystems/
echo ~~~~ > README.md
tree -d -L 2 --noreport >> README.md
cat README.md
echo " "

echo "CS3810_ComputerArchitecture"
cd ../CS3810_ComputerArchitecture/
echo ~~~~ > README.md
tree -d -L 2 --noreport >> README.md
cat README.md
echo " "

echo "CS4700_ProgrammingLanguages"
cd ../CS4700_ProgrammingLanguages/
echo ~~~~ > README.md
tree -d -L 2 --noreport >> README.md
cat README.md
echo " "

echo "ECE5930_NeuralNetworks"
cd ../ECE5930_NeuralNetworks/
echo ~~~~ > README.md
tree -d -L 2 --noreport >> README.md
cat README.md
echo " "

echo "add"
git add -u
echo "commit"
git commit -m "updated tree files"
echo "push"
git push origin master
echo -e "${GREEN}[ DONE ]${NC}"
echo " " 
