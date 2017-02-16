#include "tree.hpp"
#include <iostream>
#include <fstream>

int main(){
	std::ifstream fin;
	std::ofstream fout;
	fin.open("data.txt");
	Tree t("an elephant");
	if(fin.is_open()){
		std::cout << "......reading data to memory......" << std::endl;
		t.read(fin);
		//t.print(std::cout);
		fin.close();
	}
	else std::cout << "......no previous data......" << std::endl;
	while (isTrue("If you would like to play, think of something and hit 'y'")){
		t.start();
	}
	fout.open("data.txt");
	//t.print(std::cout);
	t.print(fout);
	fout.close();
}
