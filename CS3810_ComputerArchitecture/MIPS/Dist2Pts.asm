#########################################################################################
# Euclidian Distance Between Two Points
#########################################################################################

#########################################################################################
# DATA
#########################################################################################
	.data
s_instructions:	.asciiz	"Enter 4 integers which represent two (x, y) pairs"
s_x1:		.asciiz "x1: "
s_y1:		.asciiz "y1: "
s_x2:		.asciiz "x2: "
s_y2:		.asciiz "y2: "
s_result:	.asciiz "The distance between "
s_openParen:	.asciiz "("
s_closeParen:	.asciiz ")"
s_comma:	.asciiz ","
s_colon:	.asciiz ":"
s_space:	.asciiz " "
s_and:		.asciiz "and"
s_is:		.asciiz "is"
newline:	.asciiz "\n"

#########################################################################################
# TEXT
#########################################################################################
	.text
main:	
	la	$a0, s_instructions	# load the base of the instruction string to $a0
	jal	printsnl
	
	#################################
	# Get two points
	#################################
	
	la	$a0, s_x1		# ask for and receive x1
	jal	prints
	jal	get_int
	or	$s0, $0, $v0
	la	$a0, newline
	
	la	$a0, s_y1		# ask for and receive y1
	jal	prints
	jal	get_int
	or	$s1, $0, $v0
	la	$a0, newline
	
	la	$a0, s_x2		# ask for and receive x2
	jal	prints
	jal	get_int
	or	$s2, $0, $v0
	la	$a0, newline
	
	la	$a0, s_y2		# ask for and receive y2
	jal	prints
	jal	get_int
	or	$s3, $0, $v0
	la	$a0, newline
	
	#################################
	# Print final message
	#################################
	
	la	$a0, s_result		# print result message
	jal	prints
	
	la	$a0, s_openParen	# print (
	jal	prints
	
	or	$a0, $0, $s0		# print x1
	ori	$v0, $0, 1
	syscall
	
	la	$a0, s_comma		# print ,
	jal	prints
	la	$a0, s_space		# print ' '
	jal	prints
	
	or	$a0, $0, $s1		# print y1
	ori	$v0, $0, 1
	syscall
	
	la	$a0, s_closeParen	# print )
	jal	prints
	la	$a0, s_space		# print ' '
	jal	prints
	la	$a0, s_and		# print and
	jal	prints
	la	$a0, s_space		# print ' '
	jal	prints
	la	$a0, s_openParen	# print (
	jal	prints
	
	or	$a0, $0, $s2		# print x2
	ori	$v0, $0, 1
	syscall
	
	la	$a0, s_comma		# print ,
	jal	prints
	la	$a0, s_space		# print ' '
	jal	prints
	
	or	$a0, $0, $s3		# print y2
	ori	$v0, $0, 1
	syscall
	
	la	$a0, s_closeParen	# print )
	jal	prints
	la	$a0, s_space		# print ' '
	jal	prints
	la	$a0, s_is		# print is
	jal	prints
	la	$a0, s_colon		# print :
	jal	prints
	la	$a0, s_space		# print ' '
	jal	prints
		
	#################################
	# Calculate distance
	#################################
	
	sub	$s4, $s2, $s0		# a = x2 - x1
	sub	$s5, $s3, $s1		# b = y2 - y1
	
	mul	$s4, $s4, $s4		# a = a * a
	mul	$s5, $s5, $s5		# b = b * b
	
	add	$s6, $s4, $s5		# c = a + b
	
	or	$a0, $0, $s6		# store c in a0
	jal	sqrt			# call sqrt(a0)
	
	#################################
	# Print distance
	#################################
	
	or	$a0, $0, $v0		# put v0 in a0
	ori	$v0, $0, 1		# set syscall instruction
	syscall				# print square root
	
	j	exit			# end progaram
	
#########################################################################################
prints:
	ori	$v0, $0, 4	# load instruction to print a string
	syscall			# print string
	jr	$ra		# return
	
#########################################################################################
printsnl:
	ori	$v0, $0, 4	# load instruction to print a string
	syscall			# print string
	la	$a0, newline	# load newline
	syscall			# print newline
	jr	$ra		# return
	
#########################################################################################
get_int:
	ori	$v0, $0, 5	# load instruction to read an integer
	syscall
	jr	$ra

#########################################################################################
sqrt:	
	ori	$v0, $0, 0	# v0 = 0
	
sqrt_loop:	
	mul	$t0, $v0, $v0	# t0 = v0 * v0
	bgt	$t0, $a0, sqrt_loop_end	# if ( t0 > n ) goto end
	addi	$v0, $v0, 1	# v0 = v0 + 1
	j	sqrt_loop	# goto sqrt_loop
	
sqrt_loop_end:	
	addi	$v0, $v0, -1	# v0 = v0 - 1
	jr	$ra		# return v0


#########################################################################################
# EXIT
#########################################################################################
exit:	li $v0, 10		# load system instruction 10 (terminate)
	syscall			# terminate program
