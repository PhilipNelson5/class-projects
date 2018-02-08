#include <algorithm>
#include <iostream>
#include <sstream>

#include "wordTree.hpp"

// Determines the number of nodes it a subtree (counting the node itself).
int getSubTree(std::shared_ptr<Node> curr)
{
  if (!curr)
    return 1;
  int a = 0;
  int b = 0;
  if (curr->children)
    a = getSubTree(curr->children);
  if (curr->siblings)
    b = getSubTree(curr->siblings);

  return 1 + a + b;
}

// Recursively builds a tree given a preorder build.
std::shared_ptr<Node> LLTree::preorderBuildR(std::ifstream& fin,
                                             std::shared_ptr<Node> const parent)
{
  std::string word;
  int numChildren;

  fin >> word >> numChildren;
  auto newNode = std::make_shared<Node>(word, numChildren);

  std::shared_ptr<Node> temp;
  for (int i = 0; i < numChildren; ++i)
  {
    if (i == 0)
    {
      newNode->children = preorderBuildR(fin, newNode);
      temp = newNode->children;
    }
    else
    {
      temp->siblings = preorderBuildR(fin, newNode);
      temp = temp->siblings;
    }
  }
  newNode->subTree = getSubTree(newNode);
  newNode->parent = parent;
  return newNode;
}

// Creates a message containing the keys of a tree in preorder with the size of the sub tree
// and number of children printed next to each node.
void LLTree::printTreeR(std::shared_ptr<Node> const curr,
                        std::stringstream& ss,
                        std::string ind)
{
  if (!curr)
    return;
  ss << ind << curr->word << " [" << curr->subTree << "," << curr->numChildren << "]\n";
  printTreeR(curr->children, ss, ind + "|  ");
  printTreeR(curr->siblings, ss, ind);
}

// Returns the pointer to a node containing the given string or nullptr if not found.
std::shared_ptr<Node> LLTree::findR(std::shared_ptr<Node> curr, std::string target)
{
  if (!curr)
    return nullptr;
  if (curr->word == target)
    return curr;
  else
  {
    std::shared_ptr<Node> temp = findR(curr->children, target);
    if (!temp)
      temp = findR(curr->siblings, target);
    return temp;
  }
}

// Returns a preorder string (from a tree) in the same format expected by preorderBuild().
void LLTree::toPreorderR(std::shared_ptr<Node> curr, std::stringstream& ss) const
{
  if (!curr)
    return;
  ss << curr->word << " " << curr->numChildren << " ";
  toPreorderR(curr->children, ss);
  toPreorderR(curr->siblings, ss);
}

// Changes all the words in the tree to be uppercase. The actual tree changes.
void LLTree::upCaseR(std::shared_ptr<Node> curr)
{
  if (!curr)
    return;
  std::transform(curr->word.begin(), curr->word.end(), curr->word.begin(), toupper);
  upCaseR(curr->children);
  upCaseR(curr->siblings);
}

// Returns the count of the leaf nodes of a tree.
int LLTree::fringeR(std::shared_ptr<Node> curr, int& leaf)
{
  if (!curr)
    return 0;

  if (curr->numChildren == 0)
    ++leaf;

  if (curr->children)
    fringeR(curr->children, leaf);
  if (curr->siblings)
    fringeR(curr->siblings, leaf);

  return leaf;
}

// Produces the preorder list of the number of children to be used to compare for isomorphism.
void LLTree::getShapeR(std::shared_ptr<Node> curr, std::stringstream& ss)
{
  if (!curr)
    return;

  ss << curr->numChildren << " ";
  getShapeR(curr->children, ss);
  getShapeR(curr->siblings, ss);
}

// Compares two trees for isomorphism.
bool LLTree::isIsomorphic(LLTree other)
{
  return (this->getShape() == other.getShape());
}

// Makes a deep copy of a tree.
std::shared_ptr<Node> LLTree::cloneR(std::shared_ptr<Node> curr, std::shared_ptr<Node> parent)
{
  if (!curr)
    return nullptr;
  auto copy = std::make_shared<Node>(curr->word, curr->numChildren, curr->subTree, parent);

  copy->children = cloneR(curr->children, copy);
  copy->siblings = cloneR(curr->siblings, parent);
  return copy;
}

// Produces the list of ancestors (including the original node) to root.
void LLTree::getAncestors(std::shared_ptr<Node> curr, std::vector<std::string>& lineage)
{

  lineage.push_back(curr->word);
  if (curr->parent)
    getAncestors(curr->parent, lineage);
}

// Returns the least common ancestor of two nodes in a tree.
std::string LLTree::commonAncestor(std::string a, std::string b)
{
  std::vector<std::string> linA;
  std::vector<std::string> linB;

  auto tempA = find(a);
  auto tempB = find(b);

  if (!tempA)
    return a + " is not in the tree!\n";
  if (!tempB)
    return b + " is not in the tree!\n";

  getAncestors(tempA, linA);
  getAncestors(tempB, linB);

  for (auto&& parentA : linA)
    for (auto&& parentB : linB)
      if (parentA == parentB)
        return parentA;
  return "Common Ancestor Broke";
}

// Returns the total number of nodes on the specified level of a tree.
int LLTree::nodesInLevelR(std::shared_ptr<Node> curr, int tLevel, int cLevel, int& nodeCt)
{
  if (!curr)
    return 0;
  if (tLevel == cLevel)
    ++nodeCt;
  if (curr->children)
    nodesInLevelR(curr->children, tLevel, cLevel + 1, nodeCt);
  if (curr->siblings)
    nodesInLevelR(curr->siblings, tLevel, cLevel, nodeCt);
  return nodeCt;
}
