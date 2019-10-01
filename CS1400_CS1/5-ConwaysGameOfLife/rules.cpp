#include "rules.hpp"

int get_neighbors(std::vector<std::vector<std::pair<cell, int>>>& world, int i, int j){
	int neighbors = 0;
	/*if (j != 0 && j != world.size()-1 && i == 0){
		if (world[world.size() - 1][j + 1] == cell::ALIVE) ++neighbors;
		if (world[world.size() - 1][j] == cell::ALIVE) ++neighbors;
		if (world[world.size() - 1][j - 1] == cell::ALIVE) ++neighbors;
	}*/
	if (j != world[i].size() - 1 &&							world[  i  ][j + 1].first == cell::ALIVE) ++neighbors;
	if (j != 0 &&											world[  i  ][j - 1].first == cell::ALIVE) ++neighbors;
	if (i != world.size() - 1 && j != world[i].size() - 1 &&world[i + 1][j + 1].first == cell::ALIVE) ++neighbors;
	if (i != world.size() - 1 && 							world[i + 1][  j  ].first	== cell::ALIVE) ++neighbors;
	if (i != world.size() - 1 && j != 0 &&					world[i + 1][j - 1].first == cell::ALIVE) ++neighbors;
	if (i != 0 && j != world[i].size() - 1 &&				world[i - 1][j + 1].first == cell::ALIVE) ++neighbors;
	if (i != 0 &&											world[i - 1][  j  ].first	== cell::ALIVE) ++neighbors;
	if (i != 0 && j != 0 &&									world[i - 1][j - 1].first == cell::ALIVE) ++neighbors;
	return neighbors;	
}

cell live_die(int neighbors, cell state, std::pair<cell, int>& new_cell, int COLOR_DEPTH){
	if (state == cell::ALIVE){
		if (neighbors < 2) {
			new_cell.second = 0;
			return cell::DEAD;
		}
		else if (neighbors > 3) {
			new_cell.second = 0;
			return cell::DEAD;
		}
		else {
			new_cell.second += 1;
			return cell::ALIVE;
		}
	}
	if (state == cell::DEAD){
		if (neighbors == 3) {
			new_cell.second = 1;
			return cell::ALIVE;
		}
		else {
			new_cell.second = 0;
			return cell::DEAD;
		}
	}
	return cell::DEAD;
}

void natural_selection(std::vector<std::vector<std::pair<cell, int>>>& world, int COLOR_DEPTH){
	std::vector<std::vector<std::pair<cell, int>>> next_gen = world;
	for (unsigned int i = 0; i < world.size(); ++i)
		for (unsigned int j = 0; j < world[i].size(); ++j)
			next_gen[i][j].first = (live_die(get_neighbors(world, i, j), world[i][j].first, next_gen[i][j], COLOR_DEPTH));
	world = next_gen;
}
