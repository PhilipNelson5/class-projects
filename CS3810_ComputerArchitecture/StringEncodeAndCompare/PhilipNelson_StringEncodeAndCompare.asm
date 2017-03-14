#########################################################################################
# String Encode and Compare
#########################################################################################

#########################################################################################
# DATA
#########################################################################################
	.data
string1:	.asciiz	"Together the 2 zoos have 9 Zebras"
string2:	.asciiz	"Uphfuifs uif 3 appt ibwf 0 Afcsbt"
#string2:	.asciiz	"string two" # un-comment to test unequal strings
newline:	.asciiz "\n"
str_equal:	.asciiz "Strings are equal!"
str_nequal:	.asciiz "Strings are NOT equal!"

#########################################################################################
# TEXT
#########################################################################################
	.text
main:	la	$a1, string1	# a0 base of string1
	la	$a2, string2	# a1 base of string2

	or	$a0, $0, $a1	# load string 1 to be printed
	jal	prints

	or	$a0, $0, $a2	# load string 2 to be printed
	jal 	prints
	
	jal	strencodecmp	# encode and compare strings
	
	#or	$a0, $0, $a1	# load string 1 to be printed
	#jal	prints
	
	j	exit		# exit program
		
#########################################################################################
prints:
	ori	$v0, $0, 4	# load instruction to print a string
	syscall			# print string
	la	$a0, newline	# load newline
	syscall			# print newline
	jr	$ra		# return

#########################################################################################
printc:
	ori	$v0, $0, 11	# load instruction to print a char
	syscall			# print char
	ori	$v0, $0, 4	# load instruction to print a string
	la	$a0, newline	# load newline
	syscall			# print newline
	jr	$ra		# return
	
#########################################################################################
strencodecmp:
	sw	$ra, 0($sp)	# store return address
	addi	$sp, $sp, 4	# increment stack pointer
	
	jal	encode		# encode string
	jal	compare		# compare strings
	bnez	$v0, print_nequal # if strings are not equal jump to print_nequal

print_equal:
	la	$a0, str_equal	# load str_equal
	jal	prints		# print string
	j	print_equal_end	# skip other print
print_nequal:
	la	$a0, str_nequal	# load str_nequal
	jal	prints		# print string
print_equal_end:

	addi	$sp, $sp, -4	# decrements stack
	lw	$ra, 0($sp)	# reload return address
	jr	$ra		# return

#########################################################################################
encode:	
	ori	$t1, $0,  0	# i counter
	sw	$ra, 0($sp)	# store return address
	addi	$sp, $sp, 4	# increment stack pointer
	
encode_loop:
	add	$t2, $t1, $a1	# $t2 = &string1[i]
	lb	$t3, 0($t2)	# $t3 = string1[i]
	beq	$t3, $0, exit_encode_loop # return if ($t3 == 0) end of string
	or	$a0, $t3, $0	# load char to be encoded
	jal	encodeChar	# encode chaar in $a0
	sb	$v0, 0($t2)	# string[i] = encodeChar($t3)
	addi	$t1, $t1, 1	# increment counter
	j	encode_loop	# loop
	
exit_encode_loop:
	addi	$sp, $sp, -4	# decrements stack
	lw	$ra, 0($sp)	# reload return address
	jr	$ra		# return

#########################################################################################
encodeChar:
	beq	$a0, 90, Z	# branch if (char == 'Z')
	beq	$a0, 122, z	# branch if (char == 'z')
	beq	$a0, 57, nine	# branch if (char == 'z')
	beq	$a0, 32, space	# branch if (char == ' ')
	
	addi	$v0, $a0, 1	# normal encoding
	jr	$ra		# return
Z:	or	$v0, $0, 65	# set 'Z' to 'A'
	jr	$ra		# return
z:	or	$v0, $0, 97	# set 'z' to 'a'
	jr	$ra		# return
nine:	or	$v0, $0, 48	# set '9' to '0'
	jr	$ra		# return
space:	or	$v0, $0, 32	# leave space as space
	jr	$ra		# return
#########################################################################################
compare:
	ori	$t1, $0, 0	# i counter
	
compare_loop:
	add	$t2, $t1, $a1	# $t2 = & string1[i]
	add	$t3, $t1, $a2	# $t3 = & string2[i]
	lb	$t2, 0($t2)	# $t2 = string1[i]
	lb	$t3, 0($t3)	# $t3 = string2[i]
	
	bne	$t2, $t3, exit_compare_no # return no if($t3 != $t2) chars not equal
	beq	$t2, $0, exit_compare_yes # return yes if ($t2 == 0) end of string1
	
	addi	$t1, $t1, 1	# increment counter
	j	compare_loop

exit_compare_yes:
	ori	$v0, $0, 0	# return 0 (success)
	jr	$ra		# return
	
exit_compare_no:
	ori	$v0, $0, -1	# return -1 (failure)
	jr	$ra		# return
	
#########################################################################################
exit:	li $v0, 10		# load system instruction 10 (terminate)
	syscall			# terminate program
