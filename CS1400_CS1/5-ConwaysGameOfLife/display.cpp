#include "display.hpp"
#include "cell.hpp"

std::vector<std::vector<std::pair<cell, int>>> import(std::ifstream &fin, int IMAGE_HIGHT,
                                                      int IMAGE_WIDTH, int COLOR_DEPTH)
{
  char c;
  std::vector<std::vector<std::pair<cell, int>>> v;
  std::vector<std::pair<cell, int>> temp;
  for (int i = 0; i < IMAGE_HIGHT; ++i)
  {
    for (int j = 0; j < IMAGE_WIDTH; ++j)
    {
      fin >> c;
      if (c - '0' == 0)
        temp.push_back(std::make_pair(cell::DEAD, 0));
      else
        temp.push_back(std::make_pair(cell::ALIVE, 1));
    }
    v.push_back(temp);
    temp.clear();
  }
  return v;
}

void header(std::ostream &fout, int IMAGE_HIGHT, int IMAGE_WIDTH, int COLOR_DEPTH, int SCALE)
{
  fout << "P3" << std::endl;
  fout << IMAGE_WIDTH * SCALE << " " << IMAGE_HIGHT * SCALE << std::endl;
  fout << COLOR_DEPTH << std::endl;
}

void get_color(std::ostream &fout, std::pair<cell, int> cell)
{
  static const std::vector<int> c1{157, 220, 21};
  static const std::vector<int> c2{255, 213, 57};
  static const std::vector<int> c3{245, 143, 32};
  static const std::vector<int> c4{202, 32, 77};

  switch (cell.second)
  {
  case 1:
    fout << c1[0] << " " << c1[1] << " " << c1[2] << "\t";
    break;
  case 2:
    fout << c2[0] << " " << c2[1] << " " << c2[2] << "\t";
    break;
  case 3:
    fout << c3[0] << " " << c3[1] << " " << c3[2] << "\t";
    break;
  default:
    fout << c4[0] << " " << c4[1] << " " << c4[2] << "\t";
    break;
  }
}

void display(std::ostream &fout, std::pair<cell, int> cell, int SCALE, std::vector<int> bk_grnd)
{
  if (cell.first == cell::ALIVE)
    for (int i = 0; i < SCALE; ++i)
      get_color(fout, cell);
  else
    for (int i = 0; i < SCALE; ++i)
      fout << bk_grnd[0] << " " << bk_grnd[1] << " " << bk_grnd[2] << "\t";
}

void display(std::ostream &fout, std::vector<std::vector<std::pair<cell, int>>> v, int SCALE,
             std::vector<int> bk_grnd)
{
  for (auto &row : v)
  {
    for (int i = 0; i < SCALE; ++i)
      for (auto &cell : row)
        display(fout, cell, SCALE, bk_grnd);
    fout << std::endl;
  }
}
