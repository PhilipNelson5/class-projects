#ifndef WORLD_HPP
#define WORLD_HPP
#include "display.hpp"
#include "cell.hpp"

void simulate(std::vector<std::vector<std::pair<cell, int>>>& world, int GEN, int IMAGE_HIGHT, int IMAGE_WIDTH, int COLOR_DEPTH, int SCALE, std::string out_file_folder, std::vector<int> bk_grnd);

#endif