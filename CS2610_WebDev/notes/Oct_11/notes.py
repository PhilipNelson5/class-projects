# vim: set ft=python expandtab :
# CS 2610 - Wed Oct 11
# 
#    _                                                _      
#   /_\  _ _  _ _  ___ _  _ _ _  __ ___ _ __  ___ _ _| |_ ___
#  / _ \| ' \| ' \/ _ \ || | ' \/ _/ -_) '  \/ -_) ' \  _(_-<
# /_/ \_\_||_|_||_\___/\_,_|_||_\__\___|_|_|_\___|_||_\__/__/
#                                                            
# 
# Exam 2 will be at the beginning of next week, Mon Oct 16 - Weds 18
# 
# We'll have a review for Exam 2 on Friday, Oct 13
# 
# Due to Fall Break, there is no class on Thu Oct 19th
# 
# 
#           ___     _               _         _   _            _       
#          |_ _|_ _| |_ _ _ ___  __| |_  _ __| |_(_)___ _ _   | |_ ___ 
#           | || ' \  _| '_/ _ \/ _` | || / _|  _| / _ \ ' \  |  _/ _ \
#          |___|_||_\__|_| \___/\__,_|\_,_\__|\__|_\___/_||_|  \__\___/
#                                                                      
#                           ___      _   _             
#                          | _ \_  _| |_| |_  ___ _ _  
#                          |  _/ || |  _| ' \/ _ \ ' \ 
#                          |_|  \_, |\__|_||_\___/_||_|
#                               |__/                   
#             https://usu.instructure.com/courses/474722
# 
#
#
#  ____
# |__ /  Loops (continued)
#  |_ \  https://usu.instructure.com/courses/474722/pages/loops?module_item_id=3056235
# |___/  
# 
# * (Review) for loops iterate over collections. Examples of collections are lists and strings
# 
# * (Review) In Python, the bodies of loops are indented from their condition line
# 
# * while loops iterate so long as a condition is True.
# 
year = 1880
while (year <= 1918):
    print "The year was", year
    year += 1

print "Challenge: Fibonacci series less than 5000"
a = 0
b = 1
while (b < 5000):
    print b
    pass # TODO...
    c = b
    b = a + b
    a = c

print "Challenge: list of items, indexed"
items = ['red','orange', 'yellow', 'green']
print "length of the list",  items, "is", len(items)


#  _ _
# | | |  Control Flow
# |_  _| https://usu.instructure.com/courses/474722/pages/control-flow?module_item_id=3056254
#   |_|
#      
#  Branches are made with if, else, and elif:

score = 70
if score >= 90:
    letter = 'A'
elif score >= 80:
    letter = 'B'
elif score >= 70:
    letter = 'C'
elif score >= 60:
    letter = 'D'
else:
    letter = 'F'
print (letter)

# As with our loops, the body of a branch is set apart by indentation level.
# The condition is followed by a colon, and parentheses are optional


#  ___ 
# | __|  Types
# |__ \ 
# |___/
#
# Last time I told you that Python isn't very particular about datatypes.
#
# I don't want to leave you with the impression that Python has no concept of
# type; it certainly does have this concept.
#
# What I mean to say is that in Python variables don't have a type, but values
# do.
#
# 0, 1, 2, -3 are integers
# 0.1, 0.2, -3.3 are floats
# True, False are boolean values
#
a = 0
b = 0.1
c = False

# a, b, c are variables, and don't have types. But they values they hold do
# have types. I can assign new values of different types to existing
# variables without error:
#
a = False
b = 7
c = 3.14159

# Python's type() function can be used to query the type of a value:

type(a)
# <type 'bool'>

type(b)
# <type 'int'>

type(c)
# <type 'float'>

type(True and False)
#<type 'bool'>

type(1.0 + 77)
#<type 'float'>

# In Python the Boolean operators are spelled out
#    and, or, not
#
# The relational operators are written as in C++:
# <, <=, >, >=, ==, !=
#
# The result of an expression involving a relational operator is Boolean
type(7 < 3)
#<type 'bool'>

# Python has a special value None which is of NoneType, and works sort of like
# NULL in C++

type(None)
# <type 'NoneType'>

#   __ 
#  / /   Dictionaries
# / _ \  https://usu.instructure.com/courses/474722/pages/dictionaries?module_item_id=3057283
# \___/
#  
#  Lists are ordered collections of values. You look up items in a list by
#  their position. You can meaningfully say "item x is before item y in this
#  list".
#
#  Dictionaries, by contrast, are unordered collections of items. There is no
#  concept of "before" or "after" when it comes to dictionaries. Dictionaries
#  are key-value pairs.
#
#  Therefore, you don't store or retrieve data in a dictionary by referring to
#  its position. Instead, you give each item a name, and refer to it thus
#
#  A dictionary is denoted with curly braces { }
sculptors={
        "light_fixtures": "Elsner",
        "concentric_arcs": "Ohran",
        "pivotal_concord": "Deming",
        "french_fries": "Kinnebrew",
        "Tools_of_Ag": ["Cummings","DeGraffenried"],
        "Whispers_and_Silence": "Suzuki",
        "PrincePhraApaimanee": "Kampalanont",
        "BlockA": "Be-No Club"
        }

print sculptors['Whispers_and_Silence']
