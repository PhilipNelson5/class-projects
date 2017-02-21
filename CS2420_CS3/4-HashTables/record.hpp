#ifndef RECORD_HPP
#define RECORD_HPP

#include <iostream>
#include <sstream>
#include <string>

struct Record
{
	Record(std::string w) : word(w), count(0), probes(0) {}
	//data
	std::string word;
	//count of uses
	int count;
	//count of probes to arrive at location
	int probes;
	
	//return the word and number of times it has been used
	std::string toString()
	{
		std::ostringstream oss;
		oss << word << "\t\t" << count << "\t" << probes << std::endl;
		return oss.str();
	}

	//allows the struct to be printed via the extraciton operator
	friend std::ostream& operator<< (std::ostream& o, Record const & r)
	{
		o << r.word << "\t" << r.count << "\t" << r.probes;
		return o;
	}
};

#endif