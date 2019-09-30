.text
la $gp, GLOBAL_AREA
or $fp, $sp, $0
j MAIN
MAIN:

# number := 42;
# number, 0, 28
li $25, 42
sw $25, 0($28) # number := 42;

# write("The answer is ", number, "\n");
li $v0, 4 # load print string instruction
la $25, string0
or $a0, $0, $25 # write("The answer is ")
syscall

# number, 0, 28
li $v0, 1 # load print integer instruction
lw $25, 0($28)
or $a0, $0, $25 # write(number)
syscall

li $v0, 4 # load print string instruction
la $25, string1
or $a0, $0, $25 # write("\n")
syscall


.data
string0: .asciiz "The answer is "
string1: .asciiz "\n"

.align 2
GLOBAL_AREA:
