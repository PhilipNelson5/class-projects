#include "RandomPlayer.h"
#include <cstdlib>
#include <random>

RandomPlayer::RandomPlayer(Board::Player player):
	Player(player)
{
}

Board RandomPlayer::move(Board board)
{
	Board boards[9];
	int possible = 0;

	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
		{
			if(board(i,j)==Board::EMPTY)
			{ 
				boards[possible] = board.move(i,j,mPlayer);
				if(boards[possible].winner()==mPlayer) //if a move wins, make it
				{
					return boards[possible];
				}
				possible++;
			}
		}

	static std::random_device rd;
	static std::mt19937 mt(rd());
	std::uniform_int_distribution<> dist(1, possible);
	int choice = dist(mt);

	return boards[choice]; //choose a random move
}

