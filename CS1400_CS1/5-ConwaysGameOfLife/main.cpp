#include "cell.hpp"
#include "display.hpp"
#include "world.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main(void)
{
  // VARIABLES //
  std::ifstream fin;
  std::ofstream fout;
  int IMAGE_WIDTH, IMAGE_HIGHT, GEN, SCALE, PAUSE, COLOR_DEPTH;
  std::vector<int> background_color(3);

  // LOAD THE INITIAL STATE //
  fin.open("settings.txt");
  if (!fin)
  {
	  std::cout << "Could not read from settings.txt!" << std::endl;
	  return EXIT_FAILURE;
  }

  std::string ignore, in_file, in_file_folder, out_file_folder;

  fin >> ignore;
  fin >> GEN;
  fin >> ignore;
  fin >> SCALE;
  fin >> ignore;
  fin >> background_color[0] >> background_color[1] >> background_color[2];
  fin >> ignore;
  fin >> COLOR_DEPTH;
  fin >> ignore;
  fin >> in_file_folder;
  fin >> ignore;
  fin >> in_file;
  fin >> ignore;
  fin >> out_file_folder;
  fin >> ignore;
  fin >> PAUSE;

  fin.close();
  fin.open(in_file_folder + "/" + in_file);
  if (!fin)
  {
	  std::cout << "Could not read from " << in_file_folder + "/" + in_file << "!" << std::endl;
	  return EXIT_FAILURE;
  }

  fin >> ignore;
  fin >> IMAGE_WIDTH;
  fin >> ignore;
  fin >> IMAGE_HIGHT;

  std::cout << "IMAGE SIZE: " << IMAGE_WIDTH << " X " << IMAGE_HIGHT << std::endl;
  std::cout << "Generations: " << GEN << std::endl;
  std::cout << "Scale: " << SCALE << std::endl;

  std::vector<std::vector<std::pair<cell, int>>> world =
    import(fin, IMAGE_HIGHT, IMAGE_WIDTH, COLOR_DEPTH);
  fin.close();

  simulate(
    world, GEN, IMAGE_HIGHT, IMAGE_WIDTH, COLOR_DEPTH, SCALE, out_file_folder, background_color);

  if (PAUSE)
  {
    std::cout << "FINISHED" << std::endl << "press enter to close...";
    std::cin.get();
  }
  return EXIT_SUCCESS;
}
