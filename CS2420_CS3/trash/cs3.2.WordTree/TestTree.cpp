#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "wordTree.hpp"

int main()
{
  std::ifstream fin;
  std::ofstream fout;
  fin.open("prog2In.txt");
  fout.open("prog2Out.txt");

  if (!fin)
  {
    std::cout << "Failure to open input file :(";
    return EXIT_FAILURE;
  }
  else
    std::cout << "Input file opened" << std::endl;
  if (!fout)
  {
    std::cout << "Failure to open output file :(";
    return EXIT_FAILURE;
  }
  else
    std::cout << "Output file opened" << std::endl << std::endl;

  const int SIZE = 12;
  std::vector<LLTree> t(SIZE);

  for (int i = 0; i < SIZE; ++i)
    t[i].preorderBuild(fin);
  for (int i = 0; i < SIZE; ++i)
  {
    std::cout << "Tree" << i << std::endl << t[i].printTree();
    fout << "Tree" << i << std::endl << t[i].printTree();
    std::cout << "FRINGE " << t[i].fringe() << std::endl;
  }
  if (t[0].find("dins") == nullptr) std::cout << "dins not found" << std::endl;
  if (t[0].find("tone") == nullptr) std::cout << "tone not found" << std::endl;
  t[0].upCase();
  std::cout << t[0].printTree();
  if (t[0].find("guck") == nullptr) std::cout << "guck not found" << std::endl;
  if (t[0].find("TONE") == nullptr) std::cout << "TONE not found" << std::endl;

  t[7].makeEmpty();
  std::cout << "empty tree fringe " << t[7].fringe() << std::endl;

  for (int i = 0; i < SIZE; ++i)
  {
    std::cout << "NodesInLevel 2 of tree" << i << " " << t[i].nodesInLevel(2) << std::endl;
  }
  std::cout << " TREE 3\n" << t[3].printTree();
  std::cout << " TREE 10\n" << t[10].printTree();
  t[3] = t[10].clone();
  t[3].upCase();
  std::cout << " TREE 3 cloned\n" << t[3].printTree();
  std::cout << " TREE 10\n" << t[10].printTree();

  for (int i = 0; i < SIZE; ++i)
    for (int j = i + 1; j < SIZE; ++j)
      if (t[i].isIsomorphic(t[j]))
        std::cout << "Two trees are isomorphic Tree:" << i << " Tree:" << j << std::endl;
  std::cout << "Common Ancestor of lade and gunk " << t[2].commonAncestor("lade", "gunk")
            << std::endl;
  std::cout << "Common Ancestor of luce and gunk " << t[1].commonAncestor("luce", "gunk")
            << std::endl;
  std::cout << "Common Ancestor of lick and dene " << t[1].commonAncestor("lick", "dene")
            << std::endl;
  std::cout << "Common Ancestor of lick and muck " << t[1].commonAncestor("lick", "muck")
            << std::endl;

  fout.close();
}
