#include "TicTacToeBoard.hpp"

#include <iostream>
#include <string>

Board::Board() : winner(location_state::EMPTY)
{
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      board[i][j] = location_state::EMPTY;
}

char Board::location_state_to_char(location_state s, Player &p1, Player &p2)
{
  if (s == location_state::EMPTY) return ' ';
  if (s == location_state::ONE) return p1.get_mark();
  /*if (s == location_state::TWO)*/ return p2.get_mark();
}

void print_row(char a, char b, char c, char d, char e, char f)
{
  if (a == ' ') a = d;
  if (b == ' ') b = e;
  if (c == ' ') c = f;
  std::cout << "     |     |" << std::endl;
  std::cout << "  " << a << "  |  " << b << "  |  " << c << std::endl;
  std::cout << "     |     |" << std::endl;
}

void Board::display_board(Player p1, Player p2)
{
  print_row(location_state_to_char(board[2][0], p1, p2),
            location_state_to_char(board[2][1], p1, p2),
            location_state_to_char(board[2][2], p1, p2),
            '7',
            '8',
            '9');

  std::cout << "----- ----- -----" << std::endl;

  print_row(location_state_to_char(board[1][0], p1, p2),
            location_state_to_char(board[1][1], p1, p2),
            location_state_to_char(board[1][2], p1, p2),
            '4',
            '5',
            '6');

  std::cout << "----- ----- -----" << std::endl;

  print_row(location_state_to_char(board[0][0], p1, p2),
            location_state_to_char(board[0][1], p1, p2),
            location_state_to_char(board[0][2], p1, p2),
            '1',
            '2',
            '3');

  std::cout << std::endl;
}

bool Board::moves_left()
{
  int empty = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      if (board[i][j] == location_state::EMPTY) ++empty;
    }
  if (empty != 0) return true;
  return false;
}

bool Board::is_there_winner()
{

  /*--------------------------------[ROWS]--------------------------------*/
  if (board[0][0] == board[0][2] && board[0][1] == board[0][2] &&
      board[0][2] != location_state::EMPTY)
  { // top row
    winner = board[0][0];
    return true;
  }
  if (board[1][0] == board[1][2] && board[1][1] == board[1][2] &&
      board[1][2] != location_state::EMPTY)
  { // middle row
    winner = board[1][0];
    return true;
  }
  if (board[2][0] == board[2][2] && board[2][1] == board[2][2] &&
      board[2][2] != location_state::EMPTY)
  { // bottom row
    winner = board[2][0];
    return true;
  }

  /*--------------------------------[COLS]--------------------------------*/
  if (board[0][0] == board[2][0] && board[1][0] == board[2][0] &&
      board[2][0] != location_state::EMPTY)
  { // left col
    winner = board[0][0];
    return true;
  }
  if (board[0][1] == board[2][1] && board[1][1] == board[2][1] &&
      board[2][1] != location_state::EMPTY)
  { // middle col
    winner = board[0][1];
    return true;
  }
  if (board[0][2] == board[2][2] && board[1][2] == board[2][2] &&
      board[2][2] != location_state::EMPTY)
  { // right col
    winner = board[0][2];
    return true;
  }

  /*--------------------------------[DIAG]--------------------------------*/
  if (board[0][0] == board[2][2] && board[1][1] == board[2][2] &&
      board[2][2] != location_state::EMPTY)
  { // first diag
    winner = board[0][0];
    return true;
  }
  if (board[0][2] == board[2][0] && board[1][1] == board[2][0] &&
      board[2][0] != location_state::EMPTY)
  { // second diag
    winner = board[0][2];
    return true;
  }

  return false;
}

bool Board::is_valid_move(int move)
{
  switch (move)
  {
  case 1:
    if (board[0][0] == location_state::EMPTY)
      return true;
    else
      return false;
  case 2:
    if (board[0][1] == location_state::EMPTY)
      return true;
    else
      return false;
  case 3:
    if (board[0][2] == location_state::EMPTY)
      return true;
    else
      return false;
  case 4:
    if (board[1][0] == location_state::EMPTY)
      return true;
    else
      return false;
  case 5:
    if (board[1][1] == location_state::EMPTY)
      return true;
    else
      return false;
  case 6:
    if (board[1][2] == location_state::EMPTY)
      return true;
    else
      return false;
  case 7:
    if (board[2][0] == location_state::EMPTY)
      return true;
    else
      return false;
  case 8:
    if (board[2][1] == location_state::EMPTY)
      return true;
    else
      return false;
	default: case 9:
    if (board[2][2] == location_state::EMPTY)
      return true;
    else
      return false;
  }
}

void Board::mark_board(Player p, int move)
{

  switch (move)
  {
  case 1:
    board[0][0] = p.get_state();
    break;
  case 2:
    board[0][1] = p.get_state();
    break;
  case 3:
    board[0][2] = p.get_state();
    break;
  case 4:
    board[1][0] = p.get_state();
    break;
  case 5:
    board[1][1] = p.get_state();
    break;
  case 6:
    board[1][2] = p.get_state();
    break;
  case 7:
    board[2][0] = p.get_state();
    break;
  case 8:
    board[2][1] = p.get_state();
    break;
  case 9:
    board[2][2] = p.get_state();
    break;
  }
}

std::string Board::who_won(Player &p1, Player &p2)
{
  if (winner == location_state::ONE) return p1.get_name();
  if (winner == location_state::TWO) return p2.get_name();
  return "IT'S A TIE";
}
