// name: Philip Nelson
// A number: A01666904
// section number: 001

#include <cmath>
#include <cstdlib>
#include <iostream>

// This function determines the drawlength.
// Insert your code between the start and end comments *only*
float GetX(float distance, float theta)
{
  auto drawlength = 0.0f;
  // TODO: write your code here
  // start of your code
  const auto m = .065;              // mass (kg)
  const auto k = 25;                // spring constant (N/M)
  const auto g = 9.8;               // gravity (M/s^2)
  const auto pi = 3.14159;          // Pi
  auto thetaRad = theta * pi / 180; // theta in rads
  drawlength = (sqrt((m * g * distance) / (k * sin(2 * thetaRad))));

  // end of your code
  return drawlength;
}

// make no changes here
void testTheta(float theta, float expected)
{
  const auto dist = 100.0f;
  const auto tolerance = 0.00005f;
  auto result = GetX(dist, theta);
  auto diff = fabs(result - expected);
  std::cout << "distance: " << dist << "\ttheta: " << theta;
  std::cout << "\tdraw: " << GetX(dist, theta);
  if (diff >= tolerance)
  {
    std::cout << "\t*** TEST FAILED *** ";
  }
  std::cout << std::endl;
}

int main()
{
  testTheta(1.0f, 8.54457f);
  testTheta(15.0f, 2.25743f);
  testTheta(30.0f, 1.71528f);
  testTheta(45.0f, 1.59625f);
  testTheta(60.0f, 1.71528f);
  testTheta(75.0f, 2.25743f);
  testTheta(89.0f, 8.54428f);
  return EXIT_SUCCESS;
}
