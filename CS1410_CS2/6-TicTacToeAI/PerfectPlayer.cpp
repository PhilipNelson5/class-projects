#include "PerfectPlayer.h"
#include <algorithm>
#include <vector>

PerfectPlayer::PerfectPlayer(Board::Player player):Player(player) {}

Board::Player other(Board::Player player)
{
	if(player == Board::Player::PLAYER_X)
		return Board::Player::PLAYER_O;
	return Board::Player::PLAYER_X;
}

int PerfectPlayer::eval(Board board)
{
	if(board.winner() == mPlayer)
		return 1;
	if(board.winner() == other(mPlayer))
		return -1;
	return 0;
}

std::vector<Board> makeChildren(Board board, Board::Player player)
{
	std::vector<Board> children;
	for(int i = 0; i < 3; ++i)
	{
		for(int j = 0; j < 3; ++j)
		{
			if(board(i,j) == Board::Player::EMPTY)
				children.push_back(board.move(i, j, player));
		}
	}
	return children;
}

int PerfectPlayer::minmaxScores(Board board, bool curPlayer)
{
	if(!board.movesRemain())
		return eval(board);

	std::vector<int> scores;
	std::vector<Board> children;

	if(curPlayer)
		children = makeChildren(board, mPlayer);
	else
		children = makeChildren(board, other(mPlayer));

	for(auto && c:children)
	{
		scores.push_back(minmaxScores(c, !curPlayer));
	}

	if(curPlayer)
		return *(std::max_element(scores.begin(), scores.end()));
	else
		return *(std::min_element(scores.begin(), scores.end()));
}

Board PerfectPlayer::bestMove(Board board)
{
	std::vector<Board> children = makeChildren(board, mPlayer);
	std::vector<int> scores;

	for(auto && c : children)
	{
		scores.push_back(minmaxScores(c, false));
	}

	return children[std::distance(scores.begin(),std::max_element(scores.begin(), scores.end()))];
}

Board PerfectPlayer::move(Board board)
{
	return bestMove(board);
}

