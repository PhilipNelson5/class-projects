#include "cell.hpp"
#include "rules.hpp"
#include <SDL/SDL.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>


namespace{
	//pointers
	SDL_Window * win;
	SDL_Renderer * ren;

	//renderer location
	int x = 0;
	int y = 0;
}

void print(std::vector <std::vector <std::pair <cell, int > > > world) {
	for (auto & row : world) {
		for (auto & cell : row) {
			if (cell.first == cell::ALIVE)std::cout << "*";
			else std::cout << " ";
		}
		std::cout << std::endl;
	}
}

void increment(int WINDOW_WIDTH, int WINDOW_HIGHT) {
	if (y < WINDOW_WIDTH - 1) ++y;
	else {
		y = 0;
		if (x < WINDOW_HIGHT - 1)
			++x;
	}
}

void reset_increment() {
	x = 0; y = 0;
}

void import_settings(std::ifstream& fin, std::vector<int>& dead_color, std::vector<std::vector<int>>& live_colors,
	int& IMAGE_WIDTH, int& IMAGE_HIGHT, int& SCALE, int& COLOR_DEPTH, int& WINDOW_WIDTH, int& WINDOW_HIGHT) {

	// LOAD THE INITIAL STATE //
	std::string ignore;
	fin.open("Settings.txt");
	if (!fin)std::cout << "Could not read from file!" << std::endl;

	fin >> ignore;
	fin >> SCALE;
	fin >> ignore;
	fin >> dead_color[0] >> dead_color[1] >> dead_color[2];
	fin >> ignore;
	fin >> live_colors[0][0] >> live_colors[0][1] >> live_colors[0][2];
	fin >> ignore;
	fin >> live_colors[1][0] >> live_colors[1][1] >> live_colors[1][2];
	fin >> ignore;
	fin >> live_colors[2][0] >> live_colors[2][1] >> live_colors[2][2];
	fin >> ignore;
	fin >> live_colors[3][0] >> live_colors[3][1] >> live_colors[3][2];
	fin >> ignore;
	fin >> COLOR_DEPTH;

	fin.close();

	WINDOW_WIDTH = IMAGE_WIDTH*SCALE;
	WINDOW_HIGHT = IMAGE_HIGHT*SCALE;
}


void import_settings(std::ifstream& fin, std::vector<int>& dead_color, std::vector<std::vector<int>>& live_colors, int& IMAGE_WIDTH,
	int& IMAGE_HIGHT, int& SCALE, int& COLOR_DEPTH, int& WINDOW_WIDTH, int& WINDOW_HIGHT,
	int& DELAY, std::string& in_file, std::string& in_file_folder) {

	// LOAD THE INITIAL STATE //
	std::string ignore;
	fin.open("Settings.txt");
	if (!fin)std::cout << "Could not read from file!" << std::endl;

	fin >> ignore;
	fin >> SCALE;
	fin >> ignore;
	fin >> dead_color[0] >> dead_color[1] >> dead_color[2];
	fin >> ignore;
	fin >> live_colors[0][0] >> live_colors[0][1] >> live_colors[0][2];
	fin >> ignore;
	fin >> live_colors[1][0] >> live_colors[1][1] >> live_colors[1][2];
	fin >> ignore;
	fin >> live_colors[2][0] >> live_colors[2][1] >> live_colors[2][2];
	fin >> ignore;
	fin >> live_colors[3][0] >> live_colors[3][1] >> live_colors[3][2];
	fin >> ignore;
	fin >> COLOR_DEPTH;
	fin >> ignore;
	fin >> in_file_folder;
	fin >> ignore;
	fin >> in_file;
	fin >> ignore;
	fin >> DELAY;

	fin.close();

	fin.open(in_file_folder + "\\" + in_file);
	if (!fin)std::cout << "Could not read from file!" << std::endl;

	fin >> ignore;
	fin >> IMAGE_WIDTH;
	fin >> ignore;
	fin >> IMAGE_HIGHT;

	WINDOW_WIDTH = IMAGE_WIDTH*SCALE;
	WINDOW_HIGHT = IMAGE_HIGHT*SCALE;

	std::cout << "IMAGE SIZE: " << IMAGE_WIDTH << " X " << IMAGE_HIGHT << std::endl;
	std::cout << "WINDOW SIZE: " << WINDOW_WIDTH << " X " << WINDOW_HIGHT << std::endl;
	std::cout << "Scale: " << SCALE << std::endl;
}

std::vector<std::vector<std::pair<cell, int>>> import_initial_cond(std::ifstream& fin, int IMAGE_HIGHT, int IMAGE_WIDTH) {
	char c;
	std::vector<std::vector<std::pair<cell, int>>> v;
	std::vector<std::pair<cell, int>> temp;
	for (int i = 0; i < IMAGE_HIGHT; ++i) {
		for (int j = 0; j < IMAGE_WIDTH; ++j) {
			fin >> c;
			if (c - '0' == 0)
				temp.push_back(std::make_pair(cell::DEAD, 0));
			else temp.push_back(std::make_pair(cell::ALIVE, 1));
		}
		v.push_back(temp);
		temp.clear();
	}
	return v;
}

void get_color(std::pair<cell, int> cell, std::vector<std::vector<int>> live_colors, int WINDOW_WIDTH, int WINDOW_HIGHT) {
	switch (cell.second)
	{
	case 1:
		SDL_SetRenderDrawColor(ren, live_colors[0][0], live_colors[0][1], live_colors[0][2], SDL_ALPHA_OPAQUE);
		SDL_RenderDrawPoint(ren, y, x);
		increment(WINDOW_WIDTH, WINDOW_HIGHT);
		break;
	case 2:
		SDL_SetRenderDrawColor(ren, live_colors[1][0], live_colors[1][1], live_colors[1][2], SDL_ALPHA_OPAQUE);
		SDL_RenderDrawPoint(ren, y, x);
		increment(WINDOW_WIDTH, WINDOW_HIGHT);
		break;
	case 3:
		SDL_SetRenderDrawColor(ren, live_colors[2][0], live_colors[2][1], live_colors[2][2], SDL_ALPHA_OPAQUE);
		SDL_RenderDrawPoint(ren, y, x);
		increment(WINDOW_WIDTH, WINDOW_HIGHT);
		break;
	default:
		SDL_SetRenderDrawColor(ren, live_colors[3][0], live_colors[3][1], live_colors[3][2], SDL_ALPHA_OPAQUE);
		SDL_RenderDrawPoint(ren, y, x);
		increment(WINDOW_WIDTH, WINDOW_HIGHT);
	}
}

void display(std::pair<cell, int> cell, int SCALE, std::vector<std::vector<int>> live_colors, std::vector<int> dead_color, int WINDOW_WIDTH, int WINDOW_HIGHT) {
	if (cell.first == cell::ALIVE)
		for (int i = 0; i < SCALE; ++i)
			//assign color to image vector
			get_color(cell, live_colors, WINDOW_WIDTH, WINDOW_HIGHT);
	else
		for (int i = 0; i < SCALE; ++i) {
		SDL_SetRenderDrawColor(ren, dead_color[0], dead_color[1], dead_color[2], SDL_ALPHA_OPAQUE);
		SDL_RenderDrawPoint(ren, y, x);
		increment(WINDOW_WIDTH, WINDOW_HIGHT);
		}
}

void display(std::vector<std::vector<std::pair<cell, int>>> const & world,
	int SCALE, std::vector<std::vector<int>> live_colors, std::vector<int> dead_color, int WINDOW_WIDTH, int WINDOW_HIGHT, bool& end) {

	for (auto & row : world) {
		for (int i = 0; i < SCALE; ++i)
			for (auto & cell : row){
				display(cell, SCALE, live_colors, dead_color, WINDOW_WIDTH, WINDOW_HIGHT);
				if (cell.first == cell::ALIVE) end = false;
			}
	}
	reset_increment();
}

void Exit(){
	//desyroy window and renderer
	SDL_DestroyWindow(win);
	SDL_DestroyRenderer(ren);

	//quit SDL
	SDL_Quit();
	exit(EXIT_SUCCESS);
}

int main(int argc, char** argv) {
	/* An SDL_Event */
	SDL_Event event;
	/* A bool to check if the program has exited */
	bool quit = false;
	const int FRAMERATE = 60;
	int timerFps = 0;

	// VARIABLES //
	std::ifstream fin;
	std::ofstream fout;
	std::vector<int> dead_color(3);
	std::vector<std::vector<int>> live_colors(4);
	for (auto & row : live_colors)
		row = std::vector<int>(3);
	std::string in_file, in_file_folder;
	int IMAGE_WIDTH, IMAGE_HIGHT, SCALE, COLOR_DEPTH, WINDOW_WIDTH, WINDOW_HIGHT, DELAY;
	bool end = true;

	//import program settings
	import_settings(fin, dead_color, live_colors, IMAGE_WIDTH, IMAGE_HIGHT, SCALE, COLOR_DEPTH,
		WINDOW_WIDTH, WINDOW_HIGHT, DELAY, in_file, in_file_folder);

	//init world vector
	std::vector<std::vector<std::pair<cell, int>>> world = import_initial_cond(fin, IMAGE_HIGHT, IMAGE_WIDTH);
	fin.close();
	print(world);

	//Init SDL
	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_TIMER);

	//create window and renderer
	win = SDL_CreateWindow(in_file.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		WINDOW_WIDTH, WINDOW_HIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_RESIZABLE);
	ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

	//display generation 0
	display(world, SCALE, live_colors, dead_color, WINDOW_WIDTH, WINDOW_HIGHT, end);
	SDL_RenderPresent(ren);

	SDL_Delay(DELAY);

	while (true)/* While the program is running */
	{

		while (SDL_PollEvent(&event)) {//checks for events
			if (event.type == SDL_QUIT)/* If a quit event has been sent */
			{
				Exit();
			}
			if (event.type == SDL_MOUSEWHEEL)/* If a quit event has been sent */
			{
				if (DELAY >= 0){
					if (event.wheel.y > 0)
						DELAY += 100;
					else
						if (DELAY != 0)
							DELAY -= 100;
				}

				std::cout << DELAY << "  " << event.wheel.y << std::endl;
			}
			if (event.type == SDL_KEYDOWN)
			{
				if (event.key.keysym.sym == SDLK_ESCAPE)//escape end program
				{
					Exit();
				}
				if (event.key.keysym.sym == SDLK_r)//reloads initial conditions
				{
					import_settings(fin, dead_color, live_colors, IMAGE_WIDTH, IMAGE_HIGHT, SCALE, COLOR_DEPTH,
						WINDOW_WIDTH, WINDOW_HIGHT, DELAY, in_file, in_file_folder);


					world = import_initial_cond(fin, IMAGE_HIGHT, IMAGE_WIDTH);
					fin.close();
					SDL_SetWindowSize(win, WINDOW_WIDTH, WINDOW_HIGHT);
					print(world);
				}
				if (event.key.keysym.sym == SDLK_u)//updates conditions w/o changing current generation
				{
					import_settings(fin, dead_color, live_colors, IMAGE_WIDTH, IMAGE_HIGHT, SCALE, COLOR_DEPTH, WINDOW_WIDTH, WINDOW_HIGHT);
					SDL_SetWindowSize(win, WINDOW_WIDTH, WINDOW_HIGHT);
				}
			}
		}

		timerFps = SDL_GetTicks(); // SDL_GetTicks() reutrns the number of milliseconds since the program start.

		natural_selection(world);//next generation

		//draw current generation
		display(world, SCALE, live_colors, dead_color, WINDOW_WIDTH, WINDOW_HIGHT, end);

		timerFps = SDL_GetTicks() - timerFps; //I get the time it took to update and draw;

		if (timerFps < 1000 / FRAMERATE) // if timerFps is < 16.6666...7 ms (meaning it loaded the frame too fast)
		{
			SDL_Delay((1000 / FRAMERATE) - timerFps); //delay the frame to be in time
		}

		SDL_RenderPresent(ren);

		SDL_Delay(DELAY);
		if (end){
			import_settings(fin, dead_color, live_colors, IMAGE_WIDTH, IMAGE_HIGHT, SCALE, COLOR_DEPTH,
				WINDOW_WIDTH, WINDOW_HIGHT, DELAY, in_file, in_file_folder);
			world = import_initial_cond(fin, IMAGE_HIGHT, IMAGE_WIDTH);
			fin.close();
			SDL_SetWindowSize(win, WINDOW_WIDTH, WINDOW_HIGHT);
			print(world);
		}
		end = true;

	}
	Exit();
	return EXIT_SUCCESS;
}
