# OPsys.5.TheConchShell
A simple shell interface

Implements:

| **Command** | **Arguments** | **Description**                                                                               |
|---------|--------------|----------------------------------------------------------------------------------------------------|
| history | -            | lists all the recently entered commands.                                                           |
| ^       | *n*          | re-execute the command the *n*th command of the history.                                           |
| cd      | path/to/dir/ | change directory                                                                                   |
| ptime   | -            | prints the total amount of time spent waiting on child processes.                                  |
| color   | COLOR        | change text color ```RED     GREEN     YELLOW     BLUE     MAGENTA     CYAN     WHITE```           |
| exit    | -            | exits the ConchShell                                                                               |

aliases are also implementable via the .conchrc file. 
