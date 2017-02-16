#include <iostream>
#include <vector>

#include "Board.h"
#include "DefensiveRandomPlayer.h"
#include "HumanPlayer.h"
#include "PerfectPlayer.h"
#include "RandomPlayer.h"

using namespace std;

Player *getPlayer(Board::Player player)
{
  char choice;
  cout << "Will player ";

  if (player == Board::PLAYER_X)
  {
    cout << "X";
  }
  else
  {
    cout << "O";
  }

  cout << " be:" << endl;
  cout << "\t(H)uman" << endl;
  cout << "\t(R)andom" << endl;
  cout << "\t(D)efensive" << endl;
  cout << "\t(P)erfect" << endl;

  cin >> choice;

  switch (choice)
  {
  case 'H':
  case 'h':
    return new HumanPlayer(player);
  case 'R':
  case 'r':
    return new RandomPlayer(player);
  case 'D':
  case 'd':
    return new DefensiveRandomPlayer(player);
  default:
    return new PerfectPlayer(player);
  }
}

bool ask(std::string question)
{
  std::string ans;
  while (true)
  {
    std::cout << question << std::endl;
    std::getline(std::cin, ans);

    if (ans[0] == 'y') return true;
    if (ans[0] == 'n') return false;
    std::cout << "Enter 'y' or 'n'" << std::endl;
  }
}

int main()
{
  std::vector<int> result(3);
  Player *players[2];

  players[0] = getPlayer(Board::PLAYER_X);
  players[1] = getPlayer(Board::PLAYER_O);
  bool cont = true;
  while (cont)
  {
    Board board;
    int current_player = 0;

    while (board.movesRemain())
    {
      board = players[current_player]->move(board);
      ++current_player;
      current_player %= 2;
      board.display();
    }

    board.display();
    switch (board.winner())
    {
    case Board::PLAYER_X:
      cout << "X's Win!" << endl;
      ++result[0];
      break;

    case Board::PLAYER_O:
      cout << "O's Win!" << endl;
      ++result[1];
      break;

    case Board::EMPTY:
      cout << "Cat's Game" << endl;
      ++result[2];
      break;
    }
    cont = ask("Would you like to play again?");
  }
  cout << "X won: " << result[0] << endl
       << "O won: " << result[1] << endl
       << "cat game: " << result[2] << endl;
  delete players[0];
  delete players[1];
}
