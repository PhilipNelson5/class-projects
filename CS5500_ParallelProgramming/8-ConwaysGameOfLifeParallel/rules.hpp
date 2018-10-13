#ifndef RULES_HPP
#define RULES_HPP

#include "cell.hpp"
#include <vector>

/**
 * Count the number of neighbors for a cell in the world
 *
 * @param world The representation of the whole world
 * @param i     The row of the cell to check
 * @param j     The column of the cell to check
 * @return      The number of alive neighbors
 */
int get_neighbors(std::vector<std::vector<Cell>> const& world,
                  unsigned int i,
                  unsigned int j)
{
  int neighbors = 0;
  if (j != world[i].size() - 1 && world[i][j + 1] == Cell::ALIVE)
  {
    ++neighbors;
  }
  if (j != 0 && world[i][j - 1] == Cell::ALIVE)
  {
    ++neighbors;
  }
  if (i != world.size() - 1 && j != world[i].size() - 1 &&
      world[i + 1][j + 1] == Cell::ALIVE)
  {
    ++neighbors;
  }
  if (i != world.size() - 1 && world[i + 1][j] == Cell::ALIVE)
  {
    ++neighbors;
  }
  if (i != world.size() - 1 && j != 0 && world[i + 1][j - 1] == Cell::ALIVE)
  {
    ++neighbors;
  }
  if (i != 0 && j != world[i].size() - 1 && world[i - 1][j + 1] == Cell::ALIVE)
  {
    ++neighbors;
  }
  if (i != 0 && world[i - 1][j] == Cell::ALIVE)
  {
    ++neighbors;
  }
  if (i != 0 && j != 0 && world[i - 1][j - 1] == Cell::ALIVE)
  {
    ++neighbors;
  }
  return neighbors;
}

/**
 * determines if a cell is alive or dead based on it's current state and the
 * number of neighbors
 *
 * @param neighbors The number of living neighbors
 * @param state     The current state of a cell
 * @return          The state of the cell in the next generation
 */
Cell live_die(int neighbors, Cell state)
{
  if (state == Cell::ALIVE)
  {
    if (neighbors < 2)
    {
      return Cell::DEAD;
    }
    else if (neighbors > 3)
    {
      return Cell::DEAD;
    }
    else
    {
      return Cell::ALIVE;
    }
  }
  if (state == Cell::DEAD)
  {
    if (neighbors == 3)
    {
      return Cell::ALIVE;
    }
    else
    {
      return Cell::DEAD;
    }
  }
  return Cell::DEAD;
}

/**
 * determines the state of every cell in the world for the next generation
 *
 * @param world The representation of the whole world
 */
void natural_selection(std::vector<std::vector<Cell>>& world)
{
  std::vector<std::vector<Cell>> next_gen = world;
  for (auto i = 0u; i < world.size(); ++i)
    for (auto j = 0u; j < world[i].size(); ++j)
      next_gen[i][j] = (live_die(get_neighbors(world, i, j), world[i][j]));
  world = next_gen;
}
#endif
