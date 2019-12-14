#########################################################################################
# String Encode and Compare
#########################################################################################

#########################################################################################
# DATA
#########################################################################################
	.data
string1:	.asciiz	"The factorial of "
string2:	.asciiz	" is "

#########################################################################################
# TEXT
#########################################################################################
	.text
main:
	ori	$a1, $0, 10	# a1 will hold the number to factorial

	la	$a0, string1	# load string1
	ori	$v0, $0, 4	# load instruction to print a string
	syscall			# print string
	
	or	$a0, $0, $a1	# print number
	ori	$v0, $0, 1
	syscall
	
	la	$a0, string2	# load string1
	ori	$v0, $0, 4	# load instruction to print a string
	syscall			# print string
	
	or	$a0, $0, $a1	# call factorial func
	jal	factorial
	
	or	$a0, $0, $v0	# print result
	ori	$v0, $0, 1
	syscall

	j	exit		# exit program

#########################################################################################
# Recursively compute a factorial
#
# Registers In
#  a0 the value of the factorial to compute
#  ra the return address
#
# Registers Out
#  v0 the value of the computed factorial
#
#########################################################################################
factorial:
	beqz 	$a0, factorial_base_case# base case
	sub	$sp, $sp, 8		# move sp to save registers
	sw	$ra, 4($sp)		# save the return address
	sw	$a0, 0($sp)		# save a0
	
	sub	$a0, $a0, 1		# setup recursive call args
	jal	factorial		# call factorial
	
	lw	$a0, 0($sp)		# retreive stored registers
	lw	$ra, 4($sp)
	addi	$sp, $sp, 8
	
	mul	$v0, $v0, $a0		# return n * fact(n-1)
	jr	$ra			# return
	
factorial_base_case:
	ori	$v0, $0, 1		# factorial(0) = 1
	jr	$ra			# return
		
#########################################################################################
exit:	li $v0, 10		# load system instruction 10 (terminate)
	syscall			# terminate program