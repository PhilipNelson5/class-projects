#include <Expected/Expected.hpp>
#include <iostream>

void CHECK(bool pass)
{
    if (pass) std::cout << "PASS" << std::endl;
    else
    {
        std::cout << "FAIL" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main()
{
    Expected<int> a = 1;
    Expected<int> b = 2;
    Expected<int> c;
    CHECK(a + b == 3);
    CHECK(a - b == -1);
    CHECK(a * b == 2);
    CHECK(a / b == 0);
    CHECK(a % b == 1);
}