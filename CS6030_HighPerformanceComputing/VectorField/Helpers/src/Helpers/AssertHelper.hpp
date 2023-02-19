#pragma once

#include <cassert>

/**
 * @brief wrapper around assert macro to allow nicer syntax for adding a message
 */
#define assertm(exp, msg) assert(((void)msg, exp));
