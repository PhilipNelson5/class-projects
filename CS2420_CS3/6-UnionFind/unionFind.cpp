#include "unionFind.hpp"

bool UnionFind::isComplete() const
{
	return static_cast<unsigned int>(bigGroup) == arrH.size();
}

void UnionFind::init(int size)
{
	arrH.resize(size);
	for (auto && e : arrH)
		e = -1;

	arrS.resize(size);
	for (auto && e : arrS)
		e = -1;

	bigGroup = 0;
	finds = 0;
	touches = 0;
}

//call to union both arrays
bool UnionFind::_union(int c1, int c2)
{
	unionSize(c1, c2);
	return unionHeight(c1, c2);
}

bool UnionFind::unionHeight(int c1, int c2)
{
	int p1 = findH(c1);
	int p2 = findH(c2);
	finds += 2;
	touches += 2;

	if (p1 == p2)
		return false;

	if (arrH[p1] < arrH[p2]) //parent 1 is higher
	{
		arrH[p2] = p1;
	}
	else if (arrH[p2] < arrH[p1]) //parent 2 is higher
	{
		arrH[p1] = p2;
	}
	else //both parents are equal
	{
		--arrH[p1]; //update height
		arrH[p2] = p1;
	}

	return true;
}

bool UnionFind::unionSize(int c1, int c2)
{
	int p1 = findS(c1);
	int p2 = findS(c2);

	if (p1 == p2)
		return false;

	if (arrS[p1] < arrS[p2]) //parent 1 is larger
	{
		arrS[p1] += arrS[p2];
		arrS[p2] = p1;
		if (-arrS[p1] > bigGroup) //update size
			bigGroup = -arrS[p1];
	}
	else //(arrS[p2] <= arrS[p1]) //parten 2 is larger
	{
		arrS[p2] += arrS[p1];
		arrS[p1] = p2;
		if (-arrS[p2] > bigGroup) //update size
			bigGroup = -arrS[p2];
	}

	return true;
}
int UnionFind::findH(int num)
{
	if (arrH[num] < 0)
		return num;

	int parent = findH(arrH[num]);
	++touches;
	arrH[num] = parent;
	return parent;
}

int UnionFind::findS(int num)
{
	if (arrS[num] < 0)
		return num;

	int parent = findS(arrS[num]);
	arrS[num] = parent;
	return parent;
}
