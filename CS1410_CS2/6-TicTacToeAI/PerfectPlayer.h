#ifndef _PERFECT_PLAYER_
#define _PERFECT_PLAYER_

#include "Player.h"

class PerfectPlayer:public Player
{
	public:
		PerfectPlayer(Board::Player);
		virtual Board move(Board board);
	private:
		Board bestMove(Board); 
		int minmaxScores(Board, bool);
		int eval(Board);
};

#endif
