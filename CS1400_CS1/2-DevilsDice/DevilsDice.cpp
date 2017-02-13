// name: Philip Nelson
// A number: A01666904
// section number: 001

#include <iostream>
#include <random>
#include <string>

// variable declarations (I know it`s bad)
namespace
{
  std::string player1_name = "";
  std::string player2_name = "";
  auto player1_score = 31;
  auto player2_score = 9;
  auto game_is_over = false;
  auto again = true;
  auto graph_inc = 10;
  auto play_to = 100;
  auto die_size = 6;
  const char column_char = '|';
}

// function declarations
void welcome(); // welcomes players and displays game rules
void game();    // itteration of game
void player_turn(std::string player_name, int &player_score); // the player turn
int roll(int size);                                           // roll the dice and returns die value
void display_die(int roll);                                   // die graphic
void disp_score();                            // displays the current game score graphic
void display_winner(std::string player_name); // displays winner's banner
void mark_score(int player_score, int i);     // inserts the marks on the score graphic
void game_over();                             // checks to see if someone has won yet
bool y_n_question(
  std::string question); // ask yes or no questions and returns the answer (rejects incorrect input)
void reset();            // resets the game values in preperation for a new game
void settings();         // settings menu

int main(void)
{
  // name entry
  std::cout << "WELCOME TO THE DEVIL'S DICE!!!" << std::endl;
  std::cout << "Enter a name for player1 or type \"settings\" to enter the settings menu: ";
  std::string input = "";
  getline(std::cin, input);
  // settings option
  if (input == "settings")
  {
    settings();
    std::cout << "Enter a name for player1: ";
    std::cin.ignore();
    getline(std::cin, player1_name);
  }
  else
    player1_name = input;
  std::cout << std::endl << "Enter a name for player2 : ";
  getline(std::cin, player2_name);

  welcome();

  while (again == true)
  { // game loop
    std::cout << "Alright, Let's Play ! ! !" << std::endl;
    reset();
    game();
    again = y_n_question("Would you like to play again?");
  }
  std::cout << "Thanks for playing!" << std::endl;
  std::cout << "Application by: Philip Nelson" << std::endl;
  std::cout << "Press any key to close" << std::endl;
  std::cin.get();
  return EXIT_SUCCESS;
}

void welcome()
{ // welcomes players and displays game rules
  std::cout << std::endl
            << "Welcome " << player1_name << " and " << player2_name << "." << std::endl
            << std::endl;
  std::cout << "---------------------------------------------------------------------" << std::endl;
  std::cout << "                             [Game Rules]                            " << std::endl;
  std::cout << "---------------------------------------------------------------------" << std::endl;
  std::cout << "Each turn, a player repeatedly rolls a die until either a 1 is rolled" << std::endl
            << "   or the player decides to \"hold\"" << std::endl;
  std::cout << "If the player rolls a 1, they score nothing and it becomes the next" << std::endl
            << "   player's turn." << std::endl;
  std::cout << "If the player rolls any other number, it is added to their turn total" << std::endl
            << "   and the player's turn continues" << std::endl;
  std::cout << "If a player chooses to \"hold\", their turn total is added to their" << std::endl
            << "   score, and it becomes the next player's turn." << std::endl;
  std::cout << "The first player to score 100 or more points wins." << std::endl;
  std::cout << "Good Luck!" << std::endl << std::endl;
}

void game()
{ // itteration of game
  while (game_is_over == false)
  {
    std::cout << player1_name << " it`s your turn!" << std::endl;
    player_turn(player1_name, player1_score);
    game_over();
    if (game_is_over == true)
    {
      break;
    }
    std::cout << player2_name << " it`s your turn!" << std::endl;
    player_turn(player2_name, player2_score);
    game_over();
  }
}

void player_turn(std::string player_name, int &player_score)
{ // the player turn
  auto continue_turn = true;
  auto turn_score = 0;
  auto die_roll = 0;
  while (continue_turn == true)
  {
    if (y_n_question(player_name + " would you like to roll? "))
    {
      die_roll = roll(die_size);
      if (die_roll != 1)
      {
        display_die(die_roll);
        std::cout << "You rolled a: " << die_roll << "!" << std::endl;
        turn_score += die_roll;
      }
      else
      {
        display_die(die_roll);
        std::cout << "You rolled a 1 and gain no points this turn..." << std::endl;
        turn_score = 0;
        continue_turn = false;
      }
    }
    else
    {
      std::cout << "You won " << turn_score << " points this turn." << std::endl << std::endl;
      player_score += turn_score;
      continue_turn = false;
    }
  }
  disp_score();
}

int roll(int size)
{ // roll the dice and returns die value
  static std::random_device rd;
  static std::mt19937 mt(rd());
  std::uniform_int_distribution<> dist(1, size);
  return dist(mt);
}

void display_die(int roll)
{
  switch (roll)
  {
  case 1:
    std::cout << "-------" << std::endl;
    std::cout << "|     |" << std::endl;
    std::cout << "|  *  |" << std::endl;
    std::cout << "|     |" << std::endl;
    std::cout << "-------" << std::endl;
    break;
  case 2:
    std::cout << "-------" << std::endl;
    std::cout << "|  *  |" << std::endl;
    std::cout << "|     |" << std::endl;
    std::cout << "|  *  |" << std::endl;
    std::cout << "-------" << std::endl;
    break;
  case 3:
    std::cout << "-------" << std::endl;
    std::cout << "|*    |" << std::endl;
    std::cout << "|  *  |" << std::endl;
    std::cout << "|    *|" << std::endl;
    std::cout << "-------" << std::endl;
    break;
  case 4:
    std::cout << "-------" << std::endl;
    std::cout << "|*   *|" << std::endl;
    std::cout << "|     |" << std::endl;
    std::cout << "|*   *|" << std::endl;
    std::cout << "-------" << std::endl;
    break;
  case 5:
    std::cout << "-------" << std::endl;
    std::cout << "|*   *|" << std::endl;
    std::cout << "|  *  |" << std::endl;
    std::cout << "|*   *|" << std::endl;
    std::cout << "-------" << std::endl;
    break;
  case 6:
    std::cout << "-------" << std::endl;
    std::cout << "|*   *|" << std::endl;
    std::cout << "|*   *|" << std::endl;
    std::cout << "|*   *|" << std::endl;
    std::cout << "-------" << std::endl;
    break;
  }
}

void disp_score()
{ // displays the current game score graphic
  std::cout << "-------------------------" << std::endl;
  std::cout << "         [SCORE]         " << std::endl;
  std::cout << "-------------------------" << std::endl;
  for (int i = play_to; i >= 0; i -= graph_inc)
  {
    std::cout << "|  ";
    std::cout << i;
    mark_score(player1_score, i);
    mark_score(player2_score, i);
    std::cout << "\t|" << std::endl;
  }
  std::cout << "|_______________________|" << std::endl;
  std::cout << "|\t" << player1_name << "\t" << player2_name << "\t|" << std::endl;
  std::cout << "|-----------------------|" << std::endl;
}

void mark_score(int player_score, int i)
{ // inserts the marks on the score graphic
  std::cout << "\t";
  if (((player_score / graph_inc) * graph_inc) == i)
  {
    std::cout << player_score;
  }
  else if (player_score > i)
  {
    if (player_score < 10)
      std::cout << column_char; // single column under single digit #s
    else
      std::cout << column_char << column_char;
  }
}

void game_over()
{ // checks to see if someone has won yet
  if (player1_score >= play_to)
  {
    display_winner(player1_name);
    game_is_over = true;
  }
  else if (player2_score >= play_to)
  {
    display_winner(player2_name);
    game_is_over = true;
  }
}

void display_winner(std::string player_name)
{ // displays winner's banner
  std::cout << std::endl << "*   *   *   *   *   *   *   *   *   *" << std::endl;
  std::cout << "  *   *   *   *   *   *   *   *   *" << std::endl;
  std::cout << "   CONGRATULATIONS " << player_name << " YOU WIN!!" << std::endl;
  std::cout << "  *   *   *   *   *   *   *   *   *" << std::endl;
  std::cout << "*   *   *   *   *   *   *   *   *   *" << std::endl << std::endl;
}

bool y_n_question(std::string question)
{ // ask yes or no questions and returns the answer (rejects incorrect input)
  std::string response = "";
  char response_char = 'y';
  while (true)
  {
    std::cout << question;
    std::cout << " [Y/N]" << std::endl;
    getline(std::cin, response);
    response_char = response[0];

    if (tolower(response_char) == 'y')
      return true;
    else if (tolower(response_char) == 'n')
      return false;
    else
      std::cout << response_char << " is not a valid option..." << std::endl;
  }
}

void reset()
{ // resets the game values in preperation for a new game
  player1_score = 0;
  player2_score = 0;
  game_is_over = false;
}

void settings()
{ // settings menu
  auto setting_change = ' ';
  auto choice = 0;
  while (true)
  {
    std::cout << "[D]ice Size, [P]lay To, [G]raph Scale, [E]xit" << std::endl;
    std::cin >> setting_change;
    switch (setting_change)
    {
    case 'd':
    case 'D':
      std::cout << "How many sides on your die? " << std::endl;
      std::cin >> choice;
      die_size = choice;
      break;
    case 'p':
    case 'P':
      std::cout << "What do you want to play to?  " << std::endl;
      std::cin >> choice;
      play_to = choice;
      break;
    case 'g':
    case 'G':
      std::cout << "What do you want the scale of the score board to be?  " << std::endl;
      std::cin >> choice;
      graph_inc = choice;
      break;
    case 'e':
    case 'E':
      goto exit_loop;
    }
  }
exit_loop:;
}
