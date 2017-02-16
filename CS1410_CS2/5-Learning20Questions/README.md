# cs2.5.LetMeGuess
a learning 20 questions game

Write a program that plays a modified version of '20 questions' (There is no limit on the number of questions the computer might ask).

 

The program will ask the user to think of something and will then ask a question.

The user will respond yes or no.

Eventually the program will ask "Are you thinking of X?" with it's guess.

If the user responds with no, the program will ask "What were you thinking of?"

The user responds, then the program asks

"What question could I have asked to know that you were not thinking of Y and not X?"

The user responds with a question.

The program will update its internal state with this question to make it better at playing the game.

The program will then ask the user if it wants to play again.

 

You are not required to store the programs knowledge base.  Each run of the program will require re-training.A run of the program might look something like this (user responses in parenthesis):Think of something.Are you thinking of an elephant? (no)What were you thinking of? (a tree)

 What could I have asked to know you were thinking of a tree and not an elephant? (Are you thinking of a plant?)

Want to play again? (yes)

Think of something.

Are you thinking of a plant? (yes)

Are you thinking of a tree? (no)

What were you thinking of? (a tulip)

What could I have asked to know you were thinking of a tulip and not a tree? (Are you a flower?)

Want to play again? (yes)

Are you thinking of a plant? (no)

Are you thinking of an elephant? (yes)

I win.  Want to play again? (no)
