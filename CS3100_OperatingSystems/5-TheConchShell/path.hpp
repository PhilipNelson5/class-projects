#include <limits.h>
#include <stdio.h> /* defines FILENAME_MAX */
#include <stdlib.h>
#include <string>
#include <unistd.h>

// from: http://stackoverflow.com/questions/27914311/get-computer-name-and-logged-user-name
std::string getHostName()
{
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  return std::string(hostname);
}

std::string getLogin()
{
  return getenv("USER");
}

// from: http://www.codebind.com/cpp-tutorial/c-get-current-directory-linuxwindows/
std::string getCWD()
{
  char buff[FILENAME_MAX];
  auto cwd = getcwd(buff, FILENAME_MAX);
  std::string current_working_dir(buff);
  return (cwd != NULL) ? (std::string(current_working_dir)) : "NO DIRECTORY";
}

// http://stackoverflow.com/questions/143174/how-do-i-get-the-directory-that-a-program-is-running-from
std::string getExePath()
{
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  return std::string(result, (count > 0) ? count : 0);
}
