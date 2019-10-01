#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include "cell.hpp"
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

std::vector<std::vector<std::pair<cell, int>>> import(std::ifstream &fin, int IMAGE_HIGHT,
                                                      int IMAGE_WIDTH, int COLOR_DEPTH);

void header(std::ostream &fout, int IMAGE_HIGHT, int IMAGE_WIDTH, int COLOR_DEPTH, int SCALE);

void display(std::ostream &fout, std::vector<std::vector<std::pair<cell, int>>> v, int SCALE,
             std::vector<int> bk_grnd);

#endif
