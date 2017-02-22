#include <assert.h>
#include <iomanip>
#include <iostream>
#include "graph.hpp"

int main()
{
	std::ofstream fout;
	fout.open("tourOut.txt");
	assert(fout);

   Graph g("prog7A.txt",fout);
   g.computeTour(fout);  // If I want the output to appear on console, I just make the parameter "std::cout"

   Graph g1("prog7B.txt",fout);
   g1.computeTour(fout);

   Graph g2("prog7C.txt",cout);
   g2.computeTour(cout);
}
