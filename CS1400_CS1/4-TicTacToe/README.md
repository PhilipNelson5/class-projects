#cs1 .4.TicTacToe
Text based 2 player Tic-Tc-Toe game

The objectives of this assignment are:

1. create a header file for a class that represents a TicTacToe board
2. create an implementation file for the TicTacToe class
3. develop and implement code that uses the TicTacToe class to play the game
4. use a C++ enumeration (Section 4.13 in the book)

You start by creating a header file for a TicTacToe class. The TicTacToe class represents a TicTacToe board. You also have to create the implementation file for the TicTacToe class and develop the code to use the class to actually play a game of TicTacToe. You should use the sample output given on this page to test your program.

The main program that actually plays the TicTacToe game is not too difficult if the TicTacToe class has been implemented properly. A general pseudocode outline of the algorithm would go something like this:

1. create TicTacToe object
2. initialize variables
3. while the user wants to play another game
	1. reset the TicTacToe board
 	2. display game instructions
 	3. display the TicTacToe board
		1. while the game is still active
		2. get a valid move from the user
		3. make the move
		4. display the TicTacToe board
		5. display the game status (who won or if it was a draw)
