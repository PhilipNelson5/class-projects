#ifndef TIMER_HPP
#define TIMER_HPP

#include <algorithm>
#include <chrono>
#include <cmath>

class Timer
{
	public:
		Timer() : total(0), average(0) {}

		double total;
		std::vector<double> times;
		double average;
		std::chrono::time_point<std::chrono::system_clock> _start, _end;

		//start timer
		void start()
		{
			_start = std::chrono::high_resolution_clock::now();
		}

		//stop timer
		void end()
		{
			_end = std::chrono::high_resolution_clock::now();
			add();
		}

		//add current time to total
		void add()
		{
			total += calcTime();
		}

		//save current total
		void save()
		{
			times.push_back(total);
			total = 0;
		}

		//return time segment
		double calcTime()
		{
			return std::chrono::duration <double, std::milli>(_end - _start).count();
		}

		//return current total time
		double getTime()
		{
			return total;
		}

		//calculates the standard deviation
		double getStdDev()
		{
			double avg = getAverage();
			double dev = 0;
			for(auto && e : times)
				dev += pow((e - avg), 2);
			return sqrt(dev / times.size());
		}

		//calculates average
		double getAverage()
		{
			double sum = std::accumulate(times.begin(), times.end(), 0.0);
			return sum / times.size();
		}

		/*
		//calculates the standard deviation
		double getStdDev()
		{
		if(times.size() > 1)
		{
		double avg = getAverage();
		double dev = 0;
		for(int i = 1; i < times.size(); ++i)
		dev += pow((times[i]-avg), 2);
		return dev / (times.size() - 1);
		}
		return 0;
		}

		//calculates average
		double getAverage()
		{
		if(times.size() > 1)
		{
		double sum = 0;
		for(int i = 1; i < times.size(); ++i)
		sum += times[i];
		return sum / (times.size() - 1);
		}
		return times[0];
		}
		*/

		//resets all values
		void reset()
		{
			total = 0;
			average = 0;
			times.clear();
		}

};

/*
	 void Timer::start()

	 void Timer::end()

	 void Timer::add()

	 void Timer::save()

	 double Timer::getTime()

	 double Timer::getStdDev()

	 double Timer::getAverage()

	 void Timer::reset()
	 */

#endif
