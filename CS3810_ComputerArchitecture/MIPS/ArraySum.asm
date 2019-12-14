#########################################################################################
# Array Sum
#########################################################################################

#########################################################################################
# DATA
#########################################################################################
	.data
array:		.word   1, -2, 4, -8, 16 
string1:	.asciiz	"The sum of the array: "
string2:	.asciiz "is "


#########################################################################################
# TEXT
#########################################################################################
	.text
main:	
	la	$a0, string1	# a0 base of string1
	ori	$v0, $0, 4
	syscall
	la	$a0, array	# a0 base of array
	ori	$a1, $0, 5	# a1 is size of array
	jal	printa
	la	$a0, string2	# a0 base of string2
	ori	$v0, $0, 4
	syscall
	la	$a0, array	# a0 base of array
	ori	$a1, $0, 5	# a1 is size of array
	jal 	sum_array
	or	$a0, $0, $v0	# put return into a0
	ori	$v0, $0,  1	# load instruction 1 (print int)
	syscall			# print sum
	j	exit

#########################################################################################
# Sum the contents of an integer array
#
# registers in:
#  $a0 beginning of the array
#  $a1 number of elements in the array
#  $ra the return address
#
# registers out:
#  $v0 the sum of the array
#########################################################################################
sum_array:
	or	$t0, $0, $a0	# t0 is the base of the array
	or	$t1, $0, $0	# t1 is the counter
	or	$t2, $0, $0	# t2 is the sum
	
sum_array_loop:
	beq	$t1, $a1, sum_array_loop_end	# if (counter == size) goto loop end
	
	sll	$t5, $t1, 2	# t5 = counter * 4 (t2 = counter num of words)
	add	$t3, $t0, $t5	# t3 = &array[counter]
	lw	$t4, 0($t3)	# t4 = array[counter]
	
	add	$t2, $t2, $t4	# t2 += t4
	
	addi	$t1, $t1, 1	# ++counter
	j	sum_array_loop	# goto top of the loop

sum_array_loop_end:
	or	$v0, $0, $t2	# set return to the sum
	jr	$ra		# return

#########################################################################################
# print the contents of an integer array
#
# registers in:
#  $a0 beginning of the array
#  $a1 number of elements in the array
#  $ra the return address
#
#########################################################################################
printa:
	or	$t0, $0, $a0	# t0 is the base of the array
	or	$t1, $0, $0	# t1 is the counter
	
printa_loop:
	beq	$t1, $a1, printa_loop_end	# if (counter == size) goto loop end
	
	sll	$t2, $t1, 2	# t2 = counter * 4 (t2 = counter num of words)
	add	$t3, $t0, $t2	# t3 = &array[counter]
	lw	$a0, 0($t3)	# a0 = array[counter]
	ori	$v0, $0,  1	# load instruction 1 (print int)
	syscall			# print array[counter]
	ori	$v0, $0, 11	# load instruction 11 (print char)
	ori	$a0, $0, 44	# load 32(space) to $a0
	syscall			# print space
	ori	$a0, $0, 32	# load 32(space) to $a0
	syscall			# print space
	addi	$t1, $t1, 1	# ++counter
	j	printa_loop	# goto top of the loop
	
printa_loop_end:
	jr	$ra		# return
	
############################################################################################
exit:	li $v0, 10         	# load system instruction 10 (terminate program)
	syscall		
