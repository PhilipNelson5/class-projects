#include "world.hpp"
#include "rules.hpp"
#include <string>

void simulate(std::vector<std::vector<std::pair<cell, int>>>& world, int GEN, int IMAGE_HIGHT, int IMAGE_WIDTH, int COLOR_DEPTH, int SCALE, std::string out_file_folder, std::vector<int> bk_grnd) {
	std::string file_name;
	std::ofstream fileout("Images/generation_0.ppm");
	header(fileout, IMAGE_HIGHT, IMAGE_WIDTH, COLOR_DEPTH, SCALE);
	display(fileout, world, SCALE, bk_grnd);
	fileout.close();
	for (int i = 1; i < GEN+1; ++i) {
		file_name = out_file_folder + "/generation_" + std::to_string(i) + ".ppm";
		
		std::ofstream fout(file_name);

		header(fout, IMAGE_HIGHT, IMAGE_WIDTH, COLOR_DEPTH, SCALE);

		//simulation logic
		natural_selection(world, COLOR_DEPTH);

		display(fout, world, SCALE, bk_grnd);

		fout.close();
	}
}





/*

*/
