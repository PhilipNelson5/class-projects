#Bubble Sort
############################################################################################
# DATA
############################################################################################
	.data
#list:	.word	1,2,3,4,5,6,7,8	# sorted list
list:	.word	8,7,6,5,4,3,2,1	# unsorted list
size:	.word	8
bubble:		.asciiz		"\nBUBBLE SORT\n\n"
not_sorted:	.asciiz		"\nNOT SORTED\n"
is_sorted:	.asciiz		"\nSORTED\n"
############################################################################################
# TEXT
############################################################################################
	.text
main:	lw	$a1, size #a1 size of array
	la	$a2, list #a2 base of line array
	
	ori	$v0, $0,  4	# load instrtuction to print null terminated string
	la	$a0, bubble	# load strig to be printed
	syscall			# print string
	
	jal	print		# print list
	jal	isSorted	# check if the list is already sorted
	bnez	$v1, exit	# reads result of isSorted, exit if list already sorted
	jal	bubbleSort	# sort array
	jal	print		# print array
	jal	isSorted	# verify that list is sorted
	j	exit		# terminate program

############################################################################################
print:	
	ori	$t0, $t0, 0	# boolean done 
	ori	$t1, $0, 0	# counter
	
print_loop:
	ori	$v0, $0,  1	# load instruction 1 (print int)
	sll	$t3, $t1, 2	# counter * 4
	add	$t4, $t3, $a2	# address of list[i]
	lw	$a0, 0($t4)	# load list[i] to $a0
	syscall			# print out $a0
	ori	$v0, $0, 11	# load instruction 11 (print char)
	ori	$a0, $0, 32	# load 32(space) to $a0
	syscall			# print space
	addi	$t1, $t1, 1	# increment counter
	slt	$t0, $t1, $a1	# compare 
	bne	$t0, $0,  print_loop	# if counter != size jump to printloop
	jr	$ra
	
############################################################################################
isSorted:
	ori	$t0, $0,  0	# bool done 
	ori	$t1, $0,  0	# counter
	ori	$t3, $0,  0 	# bool inorder
	ori	$a3, $a1, 0	# coppy size
	addi	$a3, $a3, -1	# $a3 = size -1
isSorted_loop:
	sll	$t3, $t1, 2	# counter * 4
	add	$t4, $t3, $a2	# address of list[i]
	lw	$t5, 0($t4)	# list [i]
	lw	$t6, 4($t4)	# list[i+1]
	slt	$t0, $t5, $t6	# compare list[i] < list[i+1]
	beqz	$t0, exit_isSorted_no	# jump to exit_isSorted_no
	addi	$t1, $t1, 1	# increment counter
	bne	$t1, $a3, isSorted_loop	# if counter != size jump to isSorted_loop
exit_isSorted_yes:
	ori	$v1, $0,  1	# load result of isSorted 1 = true
	ori	$v0, $0,  4	# load instrtuction to print null terminated string
	la	$a0, is_sorted	# load strig to be printed
	syscall			# print string
	jr	$ra		# jump back
exit_isSorted_no:
	ori	$v1, $0,  0	# load result of isSorted 0 = false
	ori	$v0, $0,  4	# load instrtuction to print null terminated string
	la	$a0, not_sorted	# load strig to be printed
	syscall			# print string
	jr	$ra		# jump back
	
############################################################################################
bubbleSort:
	ori	$t0, $0,  0	# bool done 
	ori	$t1, $0,  0	# counter
	ori	$a3, $a1, 0	# coppy size
	addi	$a3, $a3, -1	# $a3 = size -1
	ori	$t7, $0,  0	# bool made swap
bubbleSort_loop:
	sll	$t3, $t1, 2	# counter * 4
	add	$t4, $t3, $a2	# address of list[i]
	lw	$t5, 0($t4)	# list [i]
	lw	$t6, 4($t4)	# list[i+1]
	slt	$t0, $t6, $t5	# compare list[i+1] < list[i]
	beqz	$t0, no_swap	# if not out of order dont swap
swap:
	ori	$t7, $0,  1	# set to 1 if swap
	sw	$t5, 4($t4)	# swap
	sw	$t6, 0($t4)	# swap
no_swap:
	addi	$t1, $t1, 1	# increment counter
	bne	$t1, $a3, bubbleSort_loop	# for(i <= size)
	bne	$t7, $0 , bubbleSort	# while(!done)
exit_bubbleSort:
	jr	$ra	
	
############################################################################################
exit:	li $v0, 10         # load system instruction 10 (terminate program) into v0 register
	syscall	
