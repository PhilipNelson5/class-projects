
bomb:     file format elf64-x86-64


Disassembly of section .init:

0000000000400ac0 <_init>:
  400ac0:	48 83 ec 08          	sub    $0x8,%rsp
  400ac4:	48 8b 05 2d 35 20 00 	mov    0x20352d(%rip),%rax        # 603ff8 <_DYNAMIC+0x1d0>
  400acb:	48 85 c0             	test   %rax,%rax
  400ace:	74 05                	je     400ad5 <_init+0x15>
  400ad0:	e8 cb 01 00 00       	callq  400ca0 <socket@plt+0x10>
  400ad5:	48 83 c4 08          	add    $0x8,%rsp
  400ad9:	c3                   	retq   

Disassembly of section .plt:

0000000000400ae0 <getenv@plt-0x10>:
  400ae0:	ff 35 22 35 20 00    	pushq  0x203522(%rip)        # 604008 <_GLOBAL_OFFSET_TABLE_+0x8>
  400ae6:	ff 25 24 35 20 00    	jmpq   *0x203524(%rip)        # 604010 <_GLOBAL_OFFSET_TABLE_+0x10>
  400aec:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400af0 <getenv@plt>:
  400af0:	ff 25 22 35 20 00    	jmpq   *0x203522(%rip)        # 604018 <_GLOBAL_OFFSET_TABLE_+0x18>
  400af6:	68 00 00 00 00       	pushq  $0x0
  400afb:	e9 e0 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400b00 <__errno_location@plt>:
  400b00:	ff 25 1a 35 20 00    	jmpq   *0x20351a(%rip)        # 604020 <_GLOBAL_OFFSET_TABLE_+0x20>
  400b06:	68 01 00 00 00       	pushq  $0x1
  400b0b:	e9 d0 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400b10 <strcpy@plt>:
  400b10:	ff 25 12 35 20 00    	jmpq   *0x203512(%rip)        # 604028 <_GLOBAL_OFFSET_TABLE_+0x28>
  400b16:	68 02 00 00 00       	pushq  $0x2
  400b1b:	e9 c0 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400b20 <puts@plt>:
  400b20:	ff 25 0a 35 20 00    	jmpq   *0x20350a(%rip)        # 604030 <_GLOBAL_OFFSET_TABLE_+0x30>
  400b26:	68 03 00 00 00       	pushq  $0x3
  400b2b:	e9 b0 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400b30 <write@plt>:
  400b30:	ff 25 02 35 20 00    	jmpq   *0x203502(%rip)        # 604038 <_GLOBAL_OFFSET_TABLE_+0x38>
  400b36:	68 04 00 00 00       	pushq  $0x4
  400b3b:	e9 a0 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400b40 <__stack_chk_fail@plt>:
  400b40:	ff 25 fa 34 20 00    	jmpq   *0x2034fa(%rip)        # 604040 <_GLOBAL_OFFSET_TABLE_+0x40>
  400b46:	68 05 00 00 00       	pushq  $0x5
  400b4b:	e9 90 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400b50 <alarm@plt>:
  400b50:	ff 25 f2 34 20 00    	jmpq   *0x2034f2(%rip)        # 604048 <_GLOBAL_OFFSET_TABLE_+0x48>
  400b56:	68 06 00 00 00       	pushq  $0x6
  400b5b:	e9 80 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400b60 <close@plt>:
  400b60:	ff 25 ea 34 20 00    	jmpq   *0x2034ea(%rip)        # 604050 <_GLOBAL_OFFSET_TABLE_+0x50>
  400b66:	68 07 00 00 00       	pushq  $0x7
  400b6b:	e9 70 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400b70 <read@plt>:
  400b70:	ff 25 e2 34 20 00    	jmpq   *0x2034e2(%rip)        # 604058 <_GLOBAL_OFFSET_TABLE_+0x58>
  400b76:	68 08 00 00 00       	pushq  $0x8
  400b7b:	e9 60 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400b80 <__libc_start_main@plt>:
  400b80:	ff 25 da 34 20 00    	jmpq   *0x2034da(%rip)        # 604060 <_GLOBAL_OFFSET_TABLE_+0x60>
  400b86:	68 09 00 00 00       	pushq  $0x9
  400b8b:	e9 50 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400b90 <fgets@plt>:
  400b90:	ff 25 d2 34 20 00    	jmpq   *0x2034d2(%rip)        # 604068 <_GLOBAL_OFFSET_TABLE_+0x68>
  400b96:	68 0a 00 00 00       	pushq  $0xa
  400b9b:	e9 40 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400ba0 <signal@plt>:
  400ba0:	ff 25 ca 34 20 00    	jmpq   *0x2034ca(%rip)        # 604070 <_GLOBAL_OFFSET_TABLE_+0x70>
  400ba6:	68 0b 00 00 00       	pushq  $0xb
  400bab:	e9 30 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400bb0 <gethostbyname@plt>:
  400bb0:	ff 25 c2 34 20 00    	jmpq   *0x2034c2(%rip)        # 604078 <_GLOBAL_OFFSET_TABLE_+0x78>
  400bb6:	68 0c 00 00 00       	pushq  $0xc
  400bbb:	e9 20 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400bc0 <__memmove_chk@plt>:
  400bc0:	ff 25 ba 34 20 00    	jmpq   *0x2034ba(%rip)        # 604080 <_GLOBAL_OFFSET_TABLE_+0x80>
  400bc6:	68 0d 00 00 00       	pushq  $0xd
  400bcb:	e9 10 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400bd0 <strtol@plt>:
  400bd0:	ff 25 b2 34 20 00    	jmpq   *0x2034b2(%rip)        # 604088 <_GLOBAL_OFFSET_TABLE_+0x88>
  400bd6:	68 0e 00 00 00       	pushq  $0xe
  400bdb:	e9 00 ff ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400be0 <fflush@plt>:
  400be0:	ff 25 aa 34 20 00    	jmpq   *0x2034aa(%rip)        # 604090 <_GLOBAL_OFFSET_TABLE_+0x90>
  400be6:	68 0f 00 00 00       	pushq  $0xf
  400beb:	e9 f0 fe ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400bf0 <__isoc99_sscanf@plt>:
  400bf0:	ff 25 a2 34 20 00    	jmpq   *0x2034a2(%rip)        # 604098 <_GLOBAL_OFFSET_TABLE_+0x98>
  400bf6:	68 10 00 00 00       	pushq  $0x10
  400bfb:	e9 e0 fe ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400c00 <__printf_chk@plt>:
  400c00:	ff 25 9a 34 20 00    	jmpq   *0x20349a(%rip)        # 6040a0 <_GLOBAL_OFFSET_TABLE_+0xa0>
  400c06:	68 11 00 00 00       	pushq  $0x11
  400c0b:	e9 d0 fe ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400c10 <fopen@plt>:
  400c10:	ff 25 92 34 20 00    	jmpq   *0x203492(%rip)        # 6040a8 <_GLOBAL_OFFSET_TABLE_+0xa8>
  400c16:	68 12 00 00 00       	pushq  $0x12
  400c1b:	e9 c0 fe ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400c20 <gethostname@plt>:
  400c20:	ff 25 8a 34 20 00    	jmpq   *0x20348a(%rip)        # 6040b0 <_GLOBAL_OFFSET_TABLE_+0xb0>
  400c26:	68 13 00 00 00       	pushq  $0x13
  400c2b:	e9 b0 fe ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400c30 <exit@plt>:
  400c30:	ff 25 82 34 20 00    	jmpq   *0x203482(%rip)        # 6040b8 <_GLOBAL_OFFSET_TABLE_+0xb8>
  400c36:	68 14 00 00 00       	pushq  $0x14
  400c3b:	e9 a0 fe ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400c40 <connect@plt>:
  400c40:	ff 25 7a 34 20 00    	jmpq   *0x20347a(%rip)        # 6040c0 <_GLOBAL_OFFSET_TABLE_+0xc0>
  400c46:	68 15 00 00 00       	pushq  $0x15
  400c4b:	e9 90 fe ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400c50 <__fprintf_chk@plt>:
  400c50:	ff 25 72 34 20 00    	jmpq   *0x203472(%rip)        # 6040c8 <_GLOBAL_OFFSET_TABLE_+0xc8>
  400c56:	68 16 00 00 00       	pushq  $0x16
  400c5b:	e9 80 fe ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400c60 <sleep@plt>:
  400c60:	ff 25 6a 34 20 00    	jmpq   *0x20346a(%rip)        # 6040d0 <_GLOBAL_OFFSET_TABLE_+0xd0>
  400c66:	68 17 00 00 00       	pushq  $0x17
  400c6b:	e9 70 fe ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400c70 <__ctype_b_loc@plt>:
  400c70:	ff 25 62 34 20 00    	jmpq   *0x203462(%rip)        # 6040d8 <_GLOBAL_OFFSET_TABLE_+0xd8>
  400c76:	68 18 00 00 00       	pushq  $0x18
  400c7b:	e9 60 fe ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400c80 <__sprintf_chk@plt>:
  400c80:	ff 25 5a 34 20 00    	jmpq   *0x20345a(%rip)        # 6040e0 <_GLOBAL_OFFSET_TABLE_+0xe0>
  400c86:	68 19 00 00 00       	pushq  $0x19
  400c8b:	e9 50 fe ff ff       	jmpq   400ae0 <_init+0x20>

0000000000400c90 <socket@plt>:
  400c90:	ff 25 52 34 20 00    	jmpq   *0x203452(%rip)        # 6040e8 <_GLOBAL_OFFSET_TABLE_+0xe8>
  400c96:	68 1a 00 00 00       	pushq  $0x1a
  400c9b:	e9 40 fe ff ff       	jmpq   400ae0 <_init+0x20>

Disassembly of section .plt.got:

0000000000400ca0 <.plt.got>:
  400ca0:	ff 25 52 33 20 00    	jmpq   *0x203352(%rip)        # 603ff8 <_DYNAMIC+0x1d0>
  400ca6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000400cb0 <_start>:
  400cb0:	31 ed                	xor    %ebp,%ebp
  400cb2:	49 89 d1             	mov    %rdx,%r9
  400cb5:	5e                   	pop    %rsi
  400cb6:	48 89 e2             	mov    %rsp,%rdx
  400cb9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  400cbd:	50                   	push   %rax
  400cbe:	54                   	push   %rsp
  400cbf:	49 c7 c0 50 24 40 00 	mov    $0x402450,%r8
  400cc6:	48 c7 c1 e0 23 40 00 	mov    $0x4023e0,%rcx
  400ccd:	48 c7 c7 a6 0d 40 00 	mov    $0x400da6,%rdi
  400cd4:	e8 a7 fe ff ff       	callq  400b80 <__libc_start_main@plt>
  400cd9:	f4                   	hlt    
  400cda:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400ce0 <deregister_tm_clones>:
  400ce0:	b8 87 4b 60 00       	mov    $0x604b87,%eax
  400ce5:	55                   	push   %rbp
  400ce6:	48 2d 80 4b 60 00    	sub    $0x604b80,%rax
  400cec:	48 83 f8 0e          	cmp    $0xe,%rax
  400cf0:	48 89 e5             	mov    %rsp,%rbp
  400cf3:	76 1b                	jbe    400d10 <deregister_tm_clones+0x30>
  400cf5:	b8 00 00 00 00       	mov    $0x0,%eax
  400cfa:	48 85 c0             	test   %rax,%rax
  400cfd:	74 11                	je     400d10 <deregister_tm_clones+0x30>
  400cff:	5d                   	pop    %rbp
  400d00:	bf 80 4b 60 00       	mov    $0x604b80,%edi
  400d05:	ff e0                	jmpq   *%rax
  400d07:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  400d0e:	00 00 
  400d10:	5d                   	pop    %rbp
  400d11:	c3                   	retq   
  400d12:	0f 1f 40 00          	nopl   0x0(%rax)
  400d16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  400d1d:	00 00 00 

0000000000400d20 <register_tm_clones>:
  400d20:	be 80 4b 60 00       	mov    $0x604b80,%esi
  400d25:	55                   	push   %rbp
  400d26:	48 81 ee 80 4b 60 00 	sub    $0x604b80,%rsi
  400d2d:	48 c1 fe 03          	sar    $0x3,%rsi
  400d31:	48 89 e5             	mov    %rsp,%rbp
  400d34:	48 89 f0             	mov    %rsi,%rax
  400d37:	48 c1 e8 3f          	shr    $0x3f,%rax
  400d3b:	48 01 c6             	add    %rax,%rsi
  400d3e:	48 d1 fe             	sar    %rsi
  400d41:	74 15                	je     400d58 <register_tm_clones+0x38>
  400d43:	b8 00 00 00 00       	mov    $0x0,%eax
  400d48:	48 85 c0             	test   %rax,%rax
  400d4b:	74 0b                	je     400d58 <register_tm_clones+0x38>
  400d4d:	5d                   	pop    %rbp
  400d4e:	bf 80 4b 60 00       	mov    $0x604b80,%edi
  400d53:	ff e0                	jmpq   *%rax
  400d55:	0f 1f 00             	nopl   (%rax)
  400d58:	5d                   	pop    %rbp
  400d59:	c3                   	retq   
  400d5a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000400d60 <__do_global_dtors_aux>:
  400d60:	80 3d 41 3e 20 00 00 	cmpb   $0x0,0x203e41(%rip)        # 604ba8 <completed.7585>
  400d67:	75 11                	jne    400d7a <__do_global_dtors_aux+0x1a>
  400d69:	55                   	push   %rbp
  400d6a:	48 89 e5             	mov    %rsp,%rbp
  400d6d:	e8 6e ff ff ff       	callq  400ce0 <deregister_tm_clones>
  400d72:	5d                   	pop    %rbp
  400d73:	c6 05 2e 3e 20 00 01 	movb   $0x1,0x203e2e(%rip)        # 604ba8 <completed.7585>
  400d7a:	f3 c3                	repz retq 
  400d7c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000400d80 <frame_dummy>:
  400d80:	bf 20 3e 60 00       	mov    $0x603e20,%edi
  400d85:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  400d89:	75 05                	jne    400d90 <frame_dummy+0x10>
  400d8b:	eb 93                	jmp    400d20 <register_tm_clones>
  400d8d:	0f 1f 00             	nopl   (%rax)
  400d90:	b8 00 00 00 00       	mov    $0x0,%eax
  400d95:	48 85 c0             	test   %rax,%rax
  400d98:	74 f1                	je     400d8b <frame_dummy+0xb>
  400d9a:	55                   	push   %rbp
  400d9b:	48 89 e5             	mov    %rsp,%rbp
  400d9e:	ff d0                	callq  *%rax
  400da0:	5d                   	pop    %rbp
  400da1:	e9 7a ff ff ff       	jmpq   400d20 <register_tm_clones>

0000000000400da6 <main>:
  400da6:	53                   	push   %rbx
  400da7:	83 ff 01             	cmp    $0x1,%edi
  400daa:	75 10                	jne    400dbc <main+0x16>
  400dac:	48 8b 05 dd 3d 20 00 	mov    0x203ddd(%rip),%rax        # 604b90 <stdin@@GLIBC_2.2.5>
  400db3:	48 89 05 f6 3d 20 00 	mov    %rax,0x203df6(%rip)        # 604bb0 <infile>
  400dba:	eb 63                	jmp    400e1f <main+0x79>
  400dbc:	48 89 f3             	mov    %rsi,%rbx
  400dbf:	83 ff 02             	cmp    $0x2,%edi
  400dc2:	75 3a                	jne    400dfe <main+0x58>
  400dc4:	48 8b 7e 08          	mov    0x8(%rsi),%rdi
  400dc8:	be 64 24 40 00       	mov    $0x402464,%esi
  400dcd:	e8 3e fe ff ff       	callq  400c10 <fopen@plt>
  400dd2:	48 89 05 d7 3d 20 00 	mov    %rax,0x203dd7(%rip)        # 604bb0 <infile>
  400dd9:	48 85 c0             	test   %rax,%rax
  400ddc:	75 41                	jne    400e1f <main+0x79>
  400dde:	48 8b 4b 08          	mov    0x8(%rbx),%rcx
  400de2:	48 8b 13             	mov    (%rbx),%rdx
  400de5:	be 66 24 40 00       	mov    $0x402466,%esi
  400dea:	bf 01 00 00 00       	mov    $0x1,%edi
  400def:	e8 0c fe ff ff       	callq  400c00 <__printf_chk@plt>
  400df4:	bf 08 00 00 00       	mov    $0x8,%edi
  400df9:	e8 32 fe ff ff       	callq  400c30 <exit@plt>
  400dfe:	48 8b 16             	mov    (%rsi),%rdx
  400e01:	be 83 24 40 00       	mov    $0x402483,%esi
  400e06:	bf 01 00 00 00       	mov    $0x1,%edi
  400e0b:	b8 00 00 00 00       	mov    $0x0,%eax
  400e10:	e8 eb fd ff ff       	callq  400c00 <__printf_chk@plt>
  400e15:	bf 08 00 00 00       	mov    $0x8,%edi
  400e1a:	e8 11 fe ff ff       	callq  400c30 <exit@plt>
  400e1f:	e8 be 05 00 00       	callq  4013e2 <initialize_bomb>
  400e24:	bf e8 24 40 00       	mov    $0x4024e8,%edi
  400e29:	e8 f2 fc ff ff       	callq  400b20 <puts@plt>
  400e2e:	bf 28 25 40 00       	mov    $0x402528,%edi
  400e33:	e8 e8 fc ff ff       	callq  400b20 <puts@plt>
  400e38:	e8 3e 08 00 00       	callq  40167b <read_line>
  400e3d:	48 89 c7             	mov    %rax,%rdi
  400e40:	e8 98 00 00 00       	callq  400edd <phase_1>
  400e45:	e8 57 09 00 00       	callq  4017a1 <phase_defused>
  400e4a:	bf 58 25 40 00       	mov    $0x402558,%edi
  400e4f:	e8 cc fc ff ff       	callq  400b20 <puts@plt>
  400e54:	e8 22 08 00 00       	callq  40167b <read_line>
  400e59:	48 89 c7             	mov    %rax,%rdi
  400e5c:	e8 98 00 00 00       	callq  400ef9 <phase_2>
  400e61:	e8 3b 09 00 00       	callq  4017a1 <phase_defused>
  400e66:	bf 9d 24 40 00       	mov    $0x40249d,%edi
  400e6b:	e8 b0 fc ff ff       	callq  400b20 <puts@plt>
  400e70:	e8 06 08 00 00       	callq  40167b <read_line>
  400e75:	48 89 c7             	mov    %rax,%rdi
  400e78:	e8 e8 00 00 00       	callq  400f65 <phase_3>
  400e7d:	e8 1f 09 00 00       	callq  4017a1 <phase_defused>
  400e82:	bf bb 24 40 00       	mov    $0x4024bb,%edi
  400e87:	e8 94 fc ff ff       	callq  400b20 <puts@plt>
  400e8c:	e8 ea 07 00 00       	callq  40167b <read_line>
  400e91:	48 89 c7             	mov    %rax,%rdi
  400e94:	e8 b1 01 00 00       	callq  40104a <phase_4>
  400e99:	e8 03 09 00 00       	callq  4017a1 <phase_defused>
  400e9e:	bf 88 25 40 00       	mov    $0x402588,%edi
  400ea3:	e8 78 fc ff ff       	callq  400b20 <puts@plt>
  400ea8:	e8 ce 07 00 00       	callq  40167b <read_line>
  400ead:	48 89 c7             	mov    %rax,%rdi
  400eb0:	e8 08 02 00 00       	callq  4010bd <phase_5>
  400eb5:	e8 e7 08 00 00       	callq  4017a1 <phase_defused>
  400eba:	bf ca 24 40 00       	mov    $0x4024ca,%edi
  400ebf:	e8 5c fc ff ff       	callq  400b20 <puts@plt>
  400ec4:	e8 b2 07 00 00       	callq  40167b <read_line>
  400ec9:	48 89 c7             	mov    %rax,%rdi
  400ecc:	e8 78 02 00 00       	callq  401149 <phase_6>
  400ed1:	e8 cb 08 00 00       	callq  4017a1 <phase_defused>
  400ed6:	b8 00 00 00 00       	mov    $0x0,%eax
  400edb:	5b                   	pop    %rbx
  400edc:	c3                   	retq   

0000000000400edd <phase_1>:
  400edd:	48 83 ec 08          	sub    $0x8,%rsp		# increment stack pointer
  400ee1:	be b0 25 40 00       	mov    $0x4025b0,%esi		# move value at 0x4025b0 to %esi
  400ee6:	e8 90 04 00 00       	callq  40137b <strings_not_equal>
  400eeb:	85 c0                	test   %eax,%eax
  400eed:	74 05                	je     400ef4 <phase_1+0x17>
  400eef:	e8 12 07 00 00       	callq  401606 <explode_bomb>
  400ef4:	48 83 c4 08          	add    $0x8,%rsp
  400ef8:	c3                   	retq   

0000000000400ef9 <phase_2>:
  400ef9:	55                   	push   %rbp
  400efa:	53                   	push   %rbx
  400efb:	48 83 ec 28          	sub    $0x28,%rsp
  400eff:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400f06:	00 00 
  400f08:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  400f0d:	31 c0                	xor    %eax,%eax
  400f0f:	48 89 e6             	mov    %rsp,%rsi
  400f12:	e8 25 07 00 00       	callq  40163c <read_six_numbers>
  400f17:	83 3c 24 00          	cmpl   $0x0,(%rsp)
  400f1b:	75 07                	jne    400f24 <phase_2+0x2b>
  400f1d:	83 7c 24 04 01       	cmpl   $0x1,0x4(%rsp)
  400f22:	74 05                	je     400f29 <phase_2+0x30>
  400f24:	e8 dd 06 00 00       	callq  401606 <explode_bomb>
  400f29:	48 89 e3             	mov    %rsp,%rbx
  400f2c:	48 8d 6c 24 10       	lea    0x10(%rsp),%rbp
  400f31:	8b 43 04             	mov    0x4(%rbx),%eax
  400f34:	03 03                	add    (%rbx),%eax
  400f36:	39 43 08             	cmp    %eax,0x8(%rbx)
  400f39:	74 05                	je     400f40 <phase_2+0x47>
  400f3b:	e8 c6 06 00 00       	callq  401606 <explode_bomb>
  400f40:	48 83 c3 04          	add    $0x4,%rbx
  400f44:	48 39 eb             	cmp    %rbp,%rbx
  400f47:	75 e8                	jne    400f31 <phase_2+0x38>
  400f49:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  400f4e:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400f55:	00 00 
  400f57:	74 05                	je     400f5e <phase_2+0x65>
  400f59:	e8 e2 fb ff ff       	callq  400b40 <__stack_chk_fail@plt>
  400f5e:	48 83 c4 28          	add    $0x28,%rsp
  400f62:	5b                   	pop    %rbx
  400f63:	5d                   	pop    %rbp
  400f64:	c3                   	retq   

0000000000400f65 <phase_3>:
  400f65:	48 83 ec 18          	sub    $0x18,%rsp
  400f69:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  400f70:	00 00 
  400f72:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  400f77:	31 c0                	xor    %eax,%eax
  400f79:	48 8d 4c 24 04       	lea    0x4(%rsp),%rcx
  400f7e:	48 89 e2             	mov    %rsp,%rdx
  400f81:	be b5 28 40 00       	mov    $0x4028b5,%esi
  400f86:	e8 65 fc ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  400f8b:	83 f8 01             	cmp    $0x1,%eax
  400f8e:	7f 05                	jg     400f95 <phase_3+0x30>
  400f90:	e8 71 06 00 00       	callq  401606 <explode_bomb>
  400f95:	83 3c 24 07          	cmpl   $0x7,(%rsp)
  400f99:	77 3b                	ja     400fd6 <phase_3+0x71>
  400f9b:	8b 04 24             	mov    (%rsp),%eax
  400f9e:	ff 24 c5 20 26 40 00 	jmpq   *0x402620(,%rax,8)
  400fa5:	b8 24 03 00 00       	mov    $0x324,%eax
  400faa:	eb 3b                	jmp    400fe7 <phase_3+0x82>
  400fac:	b8 ed 01 00 00       	mov    $0x1ed,%eax
  400fb1:	eb 34                	jmp    400fe7 <phase_3+0x82>
  400fb3:	b8 29 02 00 00       	mov    $0x229,%eax
  400fb8:	eb 2d                	jmp    400fe7 <phase_3+0x82>
  400fba:	b8 c6 03 00 00       	mov    $0x3c6,%eax
  400fbf:	eb 26                	jmp    400fe7 <phase_3+0x82>
  400fc1:	b8 de 03 00 00       	mov    $0x3de,%eax
  400fc6:	eb 1f                	jmp    400fe7 <phase_3+0x82>
  400fc8:	b8 f1 02 00 00       	mov    $0x2f1,%eax
  400fcd:	eb 18                	jmp    400fe7 <phase_3+0x82>
  400fcf:	b8 99 00 00 00       	mov    $0x99,%eax
  400fd4:	eb 11                	jmp    400fe7 <phase_3+0x82>
  400fd6:	e8 2b 06 00 00       	callq  401606 <explode_bomb>
  400fdb:	b8 00 00 00 00       	mov    $0x0,%eax
  400fe0:	eb 05                	jmp    400fe7 <phase_3+0x82>
  400fe2:	b8 98 03 00 00       	mov    $0x398,%eax
  400fe7:	3b 44 24 04          	cmp    0x4(%rsp),%eax
  400feb:	74 05                	je     400ff2 <phase_3+0x8d>
  400fed:	e8 14 06 00 00       	callq  401606 <explode_bomb>
  400ff2:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  400ff7:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  400ffe:	00 00 
  401000:	74 05                	je     401007 <phase_3+0xa2>
  401002:	e8 39 fb ff ff       	callq  400b40 <__stack_chk_fail@plt>
  401007:	48 83 c4 18          	add    $0x18,%rsp
  40100b:	c3                   	retq   

000000000040100c <func4>:
  40100c:	48 83 ec 08          	sub    $0x8,%rsp
  401010:	89 d0                	mov    %edx,%eax
  401012:	29 f0                	sub    %esi,%eax
  401014:	89 c1                	mov    %eax,%ecx
  401016:	c1 e9 1f             	shr    $0x1f,%ecx
  401019:	01 c8                	add    %ecx,%eax
  40101b:	d1 f8                	sar    %eax
  40101d:	8d 0c 30             	lea    (%rax,%rsi,1),%ecx
  401020:	39 f9                	cmp    %edi,%ecx
  401022:	7e 0c                	jle    401030 <func4+0x24>
  401024:	8d 51 ff             	lea    -0x1(%rcx),%edx
  401027:	e8 e0 ff ff ff       	callq  40100c <func4>
  40102c:	01 c0                	add    %eax,%eax
  40102e:	eb 15                	jmp    401045 <func4+0x39>
  401030:	b8 00 00 00 00       	mov    $0x0,%eax
  401035:	39 f9                	cmp    %edi,%ecx
  401037:	7d 0c                	jge    401045 <func4+0x39>
  401039:	8d 71 01             	lea    0x1(%rcx),%esi
  40103c:	e8 cb ff ff ff       	callq  40100c <func4>
  401041:	8d 44 00 01          	lea    0x1(%rax,%rax,1),%eax
  401045:	48 83 c4 08          	add    $0x8,%rsp
  401049:	c3                   	retq   

000000000040104a <phase_4>:
  40104a:	48 83 ec 18          	sub    $0x18,%rsp
  40104e:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401055:	00 00 
  401057:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40105c:	31 c0                	xor    %eax,%eax
  40105e:	48 8d 4c 24 04       	lea    0x4(%rsp),%rcx
  401063:	48 89 e2             	mov    %rsp,%rdx
  401066:	be b5 28 40 00       	mov    $0x4028b5,%esi
  40106b:	e8 80 fb ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  401070:	83 f8 02             	cmp    $0x2,%eax
  401073:	75 06                	jne    40107b <phase_4+0x31>
  401075:	83 3c 24 0e          	cmpl   $0xe,(%rsp)
  401079:	76 05                	jbe    401080 <phase_4+0x36>
  40107b:	e8 86 05 00 00       	callq  401606 <explode_bomb>
  401080:	ba 0e 00 00 00       	mov    $0xe,%edx
  401085:	be 00 00 00 00       	mov    $0x0,%esi
  40108a:	8b 3c 24             	mov    (%rsp),%edi
  40108d:	e8 7a ff ff ff       	callq  40100c <func4>
  401092:	83 f8 03             	cmp    $0x3,%eax
  401095:	75 07                	jne    40109e <phase_4+0x54>
  401097:	83 7c 24 04 03       	cmpl   $0x3,0x4(%rsp)
  40109c:	74 05                	je     4010a3 <phase_4+0x59>
  40109e:	e8 63 05 00 00       	callq  401606 <explode_bomb>
  4010a3:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  4010a8:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4010af:	00 00 
  4010b1:	74 05                	je     4010b8 <phase_4+0x6e>
  4010b3:	e8 88 fa ff ff       	callq  400b40 <__stack_chk_fail@plt>
  4010b8:	48 83 c4 18          	add    $0x18,%rsp
  4010bc:	c3                   	retq   

00000000004010bd <phase_5>:
  4010bd:	48 83 ec 18          	sub    $0x18,%rsp
  4010c1:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4010c8:	00 00 
  4010ca:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4010cf:	31 c0                	xor    %eax,%eax
  4010d1:	48 8d 4c 24 04       	lea    0x4(%rsp),%rcx
  4010d6:	48 89 e2             	mov    %rsp,%rdx
  4010d9:	be b5 28 40 00       	mov    $0x4028b5,%esi
  4010de:	e8 0d fb ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  4010e3:	83 f8 01             	cmp    $0x1,%eax
  4010e6:	7f 05                	jg     4010ed <phase_5+0x30>
  4010e8:	e8 19 05 00 00       	callq  401606 <explode_bomb>
  4010ed:	8b 04 24             	mov    (%rsp),%eax
  4010f0:	83 e0 0f             	and    $0xf,%eax
  4010f3:	89 04 24             	mov    %eax,(%rsp)
  4010f6:	83 f8 0f             	cmp    $0xf,%eax
  4010f9:	74 2f                	je     40112a <phase_5+0x6d>
  4010fb:	b9 00 00 00 00       	mov    $0x0,%ecx
  401100:	ba 00 00 00 00       	mov    $0x0,%edx
  401105:	83 c2 01             	add    $0x1,%edx
  401108:	48 98                	cltq   
  40110a:	8b 04 85 60 26 40 00 	mov    0x402660(,%rax,4),%eax
  401111:	01 c1                	add    %eax,%ecx
  401113:	83 f8 0f             	cmp    $0xf,%eax
  401116:	75 ed                	jne    401105 <phase_5+0x48>
  401118:	c7 04 24 0f 00 00 00 	movl   $0xf,(%rsp)
  40111f:	83 fa 0f             	cmp    $0xf,%edx
  401122:	75 06                	jne    40112a <phase_5+0x6d>
  401124:	3b 4c 24 04          	cmp    0x4(%rsp),%ecx
  401128:	74 05                	je     40112f <phase_5+0x72>
  40112a:	e8 d7 04 00 00       	callq  401606 <explode_bomb>
  40112f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  401134:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40113b:	00 00 
  40113d:	74 05                	je     401144 <phase_5+0x87>
  40113f:	e8 fc f9 ff ff       	callq  400b40 <__stack_chk_fail@plt>
  401144:	48 83 c4 18          	add    $0x18,%rsp
  401148:	c3                   	retq   

0000000000401149 <phase_6>:
  401149:	41 55                	push   %r13
  40114b:	41 54                	push   %r12
  40114d:	55                   	push   %rbp
  40114e:	53                   	push   %rbx
  40114f:	48 83 ec 68          	sub    $0x68,%rsp
  401153:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40115a:	00 00 
  40115c:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
  401161:	31 c0                	xor    %eax,%eax
  401163:	48 89 e6             	mov    %rsp,%rsi
  401166:	e8 d1 04 00 00       	callq  40163c <read_six_numbers>
  40116b:	49 89 e4             	mov    %rsp,%r12
  40116e:	41 bd 00 00 00 00    	mov    $0x0,%r13d
  401174:	4c 89 e5             	mov    %r12,%rbp
  401177:	41 8b 04 24          	mov    (%r12),%eax
  40117b:	83 e8 01             	sub    $0x1,%eax
  40117e:	83 f8 05             	cmp    $0x5,%eax
  401181:	76 05                	jbe    401188 <phase_6+0x3f>
  401183:	e8 7e 04 00 00       	callq  401606 <explode_bomb>
  401188:	41 83 c5 01          	add    $0x1,%r13d
  40118c:	41 83 fd 06          	cmp    $0x6,%r13d
  401190:	74 3d                	je     4011cf <phase_6+0x86>
  401192:	44 89 eb             	mov    %r13d,%ebx
  401195:	48 63 c3             	movslq %ebx,%rax
  401198:	8b 04 84             	mov    (%rsp,%rax,4),%eax
  40119b:	39 45 00             	cmp    %eax,0x0(%rbp)
  40119e:	75 05                	jne    4011a5 <phase_6+0x5c>
  4011a0:	e8 61 04 00 00       	callq  401606 <explode_bomb>
  4011a5:	83 c3 01             	add    $0x1,%ebx
  4011a8:	83 fb 05             	cmp    $0x5,%ebx
  4011ab:	7e e8                	jle    401195 <phase_6+0x4c>
  4011ad:	49 83 c4 04          	add    $0x4,%r12
  4011b1:	eb c1                	jmp    401174 <phase_6+0x2b>
  4011b3:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  4011b7:	83 c0 01             	add    $0x1,%eax
  4011ba:	39 c8                	cmp    %ecx,%eax
  4011bc:	75 f5                	jne    4011b3 <phase_6+0x6a>
  4011be:	48 89 54 74 20       	mov    %rdx,0x20(%rsp,%rsi,2)
  4011c3:	48 83 c6 04          	add    $0x4,%rsi
  4011c7:	48 83 fe 18          	cmp    $0x18,%rsi
  4011cb:	75 07                	jne    4011d4 <phase_6+0x8b>
  4011cd:	eb 19                	jmp    4011e8 <phase_6+0x9f>
  4011cf:	be 00 00 00 00       	mov    $0x0,%esi
  4011d4:	8b 0c 34             	mov    (%rsp,%rsi,1),%ecx
  4011d7:	b8 01 00 00 00       	mov    $0x1,%eax
  4011dc:	ba 00 43 60 00       	mov    $0x604300,%edx
  4011e1:	83 f9 01             	cmp    $0x1,%ecx
  4011e4:	7f cd                	jg     4011b3 <phase_6+0x6a>
  4011e6:	eb d6                	jmp    4011be <phase_6+0x75>
  4011e8:	48 8b 5c 24 20       	mov    0x20(%rsp),%rbx
  4011ed:	48 8d 44 24 20       	lea    0x20(%rsp),%rax
  4011f2:	48 8d 74 24 48       	lea    0x48(%rsp),%rsi
  4011f7:	48 89 d9             	mov    %rbx,%rcx
  4011fa:	48 8b 50 08          	mov    0x8(%rax),%rdx
  4011fe:	48 89 51 08          	mov    %rdx,0x8(%rcx)
  401202:	48 83 c0 08          	add    $0x8,%rax
  401206:	48 89 d1             	mov    %rdx,%rcx
  401209:	48 39 f0             	cmp    %rsi,%rax
  40120c:	75 ec                	jne    4011fa <phase_6+0xb1>
  40120e:	48 c7 42 08 00 00 00 	movq   $0x0,0x8(%rdx)
  401215:	00 
  401216:	bd 05 00 00 00       	mov    $0x5,%ebp
  40121b:	48 8b 43 08          	mov    0x8(%rbx),%rax
  40121f:	8b 00                	mov    (%rax),%eax
  401221:	39 03                	cmp    %eax,(%rbx)
  401223:	7e 05                	jle    40122a <phase_6+0xe1>
  401225:	e8 dc 03 00 00       	callq  401606 <explode_bomb>
  40122a:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
  40122e:	83 ed 01             	sub    $0x1,%ebp
  401231:	75 e8                	jne    40121b <phase_6+0xd2>
  401233:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
  401238:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40123f:	00 00 
  401241:	74 05                	je     401248 <phase_6+0xff>
  401243:	e8 f8 f8 ff ff       	callq  400b40 <__stack_chk_fail@plt>
  401248:	48 83 c4 68          	add    $0x68,%rsp
  40124c:	5b                   	pop    %rbx
  40124d:	5d                   	pop    %rbp
  40124e:	41 5c                	pop    %r12
  401250:	41 5d                	pop    %r13
  401252:	c3                   	retq   

0000000000401253 <fun7>:
  401253:	48 83 ec 08          	sub    $0x8,%rsp
  401257:	48 85 ff             	test   %rdi,%rdi
  40125a:	74 2b                	je     401287 <fun7+0x34>
  40125c:	8b 17                	mov    (%rdi),%edx
  40125e:	39 f2                	cmp    %esi,%edx
  401260:	7e 0d                	jle    40126f <fun7+0x1c>
  401262:	48 8b 7f 08          	mov    0x8(%rdi),%rdi
  401266:	e8 e8 ff ff ff       	callq  401253 <fun7>
  40126b:	01 c0                	add    %eax,%eax
  40126d:	eb 1d                	jmp    40128c <fun7+0x39>
  40126f:	b8 00 00 00 00       	mov    $0x0,%eax
  401274:	39 f2                	cmp    %esi,%edx
  401276:	74 14                	je     40128c <fun7+0x39>
  401278:	48 8b 7f 10          	mov    0x10(%rdi),%rdi
  40127c:	e8 d2 ff ff ff       	callq  401253 <fun7>
  401281:	8d 44 00 01          	lea    0x1(%rax,%rax,1),%eax
  401285:	eb 05                	jmp    40128c <fun7+0x39>
  401287:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40128c:	48 83 c4 08          	add    $0x8,%rsp
  401290:	c3                   	retq   

0000000000401291 <secret_phase>:
  401291:	53                   	push   %rbx
  401292:	e8 e4 03 00 00       	callq  40167b <read_line>
  401297:	ba 0a 00 00 00       	mov    $0xa,%edx
  40129c:	be 00 00 00 00       	mov    $0x0,%esi
  4012a1:	48 89 c7             	mov    %rax,%rdi
  4012a4:	e8 27 f9 ff ff       	callq  400bd0 <strtol@plt>
  4012a9:	48 89 c3             	mov    %rax,%rbx
  4012ac:	8d 40 ff             	lea    -0x1(%rax),%eax
  4012af:	3d e8 03 00 00       	cmp    $0x3e8,%eax
  4012b4:	76 05                	jbe    4012bb <secret_phase+0x2a>
  4012b6:	e8 4b 03 00 00       	callq  401606 <explode_bomb>
  4012bb:	89 de                	mov    %ebx,%esi
  4012bd:	bf 20 41 60 00       	mov    $0x604120,%edi
  4012c2:	e8 8c ff ff ff       	callq  401253 <fun7>
  4012c7:	83 f8 02             	cmp    $0x2,%eax
  4012ca:	74 05                	je     4012d1 <secret_phase+0x40>
  4012cc:	e8 35 03 00 00       	callq  401606 <explode_bomb>
  4012d1:	bf e0 25 40 00       	mov    $0x4025e0,%edi
  4012d6:	e8 45 f8 ff ff       	callq  400b20 <puts@plt>
  4012db:	e8 c1 04 00 00       	callq  4017a1 <phase_defused>
  4012e0:	5b                   	pop    %rbx
  4012e1:	c3                   	retq   

00000000004012e2 <sig_handler>:
  4012e2:	48 83 ec 08          	sub    $0x8,%rsp
  4012e6:	bf a0 26 40 00       	mov    $0x4026a0,%edi
  4012eb:	e8 30 f8 ff ff       	callq  400b20 <puts@plt>
  4012f0:	bf 03 00 00 00       	mov    $0x3,%edi
  4012f5:	e8 66 f9 ff ff       	callq  400c60 <sleep@plt>
  4012fa:	be 31 28 40 00       	mov    $0x402831,%esi
  4012ff:	bf 01 00 00 00       	mov    $0x1,%edi
  401304:	b8 00 00 00 00       	mov    $0x0,%eax
  401309:	e8 f2 f8 ff ff       	callq  400c00 <__printf_chk@plt>
  40130e:	48 8b 3d 6b 38 20 00 	mov    0x20386b(%rip),%rdi        # 604b80 <__TMC_END__>
  401315:	e8 c6 f8 ff ff       	callq  400be0 <fflush@plt>
  40131a:	bf 01 00 00 00       	mov    $0x1,%edi
  40131f:	e8 3c f9 ff ff       	callq  400c60 <sleep@plt>
  401324:	bf 39 28 40 00       	mov    $0x402839,%edi
  401329:	e8 f2 f7 ff ff       	callq  400b20 <puts@plt>
  40132e:	bf 10 00 00 00       	mov    $0x10,%edi
  401333:	e8 f8 f8 ff ff       	callq  400c30 <exit@plt>

0000000000401338 <invalid_phase>:
  401338:	48 83 ec 08          	sub    $0x8,%rsp
  40133c:	48 89 fa             	mov    %rdi,%rdx
  40133f:	be 41 28 40 00       	mov    $0x402841,%esi
  401344:	bf 01 00 00 00       	mov    $0x1,%edi
  401349:	b8 00 00 00 00       	mov    $0x0,%eax
  40134e:	e8 ad f8 ff ff       	callq  400c00 <__printf_chk@plt>
  401353:	bf 08 00 00 00       	mov    $0x8,%edi
  401358:	e8 d3 f8 ff ff       	callq  400c30 <exit@plt>

000000000040135d <string_length>:
  40135d:	80 3f 00             	cmpb   $0x0,(%rdi)
  401360:	74 13                	je     401375 <string_length+0x18>
  401362:	b8 00 00 00 00       	mov    $0x0,%eax
  401367:	48 83 c7 01          	add    $0x1,%rdi
  40136b:	83 c0 01             	add    $0x1,%eax
  40136e:	80 3f 00             	cmpb   $0x0,(%rdi)
  401371:	75 f4                	jne    401367 <string_length+0xa>
  401373:	f3 c3                	repz retq 
  401375:	b8 00 00 00 00       	mov    $0x0,%eax
  40137a:	c3                   	retq   

000000000040137b <strings_not_equal>:
  40137b:	41 54                	push   %r12
  40137d:	55                   	push   %rbp
  40137e:	53                   	push   %rbx
  40137f:	48 89 fb             	mov    %rdi,%rbx
  401382:	48 89 f5             	mov    %rsi,%rbp
  401385:	e8 d3 ff ff ff       	callq  40135d <string_length>
  40138a:	41 89 c4             	mov    %eax,%r12d
  40138d:	48 89 ef             	mov    %rbp,%rdi
  401390:	e8 c8 ff ff ff       	callq  40135d <string_length>
  401395:	ba 01 00 00 00       	mov    $0x1,%edx
  40139a:	41 39 c4             	cmp    %eax,%r12d
  40139d:	75 3c                	jne    4013db <strings_not_equal+0x60>
  40139f:	0f b6 03             	movzbl (%rbx),%eax
  4013a2:	84 c0                	test   %al,%al
  4013a4:	74 22                	je     4013c8 <strings_not_equal+0x4d>
  4013a6:	3a 45 00             	cmp    0x0(%rbp),%al
  4013a9:	74 07                	je     4013b2 <strings_not_equal+0x37>
  4013ab:	eb 22                	jmp    4013cf <strings_not_equal+0x54>
  4013ad:	3a 45 00             	cmp    0x0(%rbp),%al
  4013b0:	75 24                	jne    4013d6 <strings_not_equal+0x5b>
  4013b2:	48 83 c3 01          	add    $0x1,%rbx
  4013b6:	48 83 c5 01          	add    $0x1,%rbp
  4013ba:	0f b6 03             	movzbl (%rbx),%eax
  4013bd:	84 c0                	test   %al,%al
  4013bf:	75 ec                	jne    4013ad <strings_not_equal+0x32>
  4013c1:	ba 00 00 00 00       	mov    $0x0,%edx
  4013c6:	eb 13                	jmp    4013db <strings_not_equal+0x60>
  4013c8:	ba 00 00 00 00       	mov    $0x0,%edx
  4013cd:	eb 0c                	jmp    4013db <strings_not_equal+0x60>
  4013cf:	ba 01 00 00 00       	mov    $0x1,%edx
  4013d4:	eb 05                	jmp    4013db <strings_not_equal+0x60>
  4013d6:	ba 01 00 00 00       	mov    $0x1,%edx
  4013db:	89 d0                	mov    %edx,%eax
  4013dd:	5b                   	pop    %rbx
  4013de:	5d                   	pop    %rbp
  4013df:	41 5c                	pop    %r12
  4013e1:	c3                   	retq   

00000000004013e2 <initialize_bomb>:
  4013e2:	48 81 ec 58 20 00 00 	sub    $0x2058,%rsp
  4013e9:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4013f0:	00 00 
  4013f2:	48 89 84 24 48 20 00 	mov    %rax,0x2048(%rsp)
  4013f9:	00 
  4013fa:	31 c0                	xor    %eax,%eax
  4013fc:	be e2 12 40 00       	mov    $0x4012e2,%esi
  401401:	bf 02 00 00 00       	mov    $0x2,%edi
  401406:	e8 95 f7 ff ff       	callq  400ba0 <signal@plt>
  40140b:	be 40 00 00 00       	mov    $0x40,%esi
  401410:	48 89 e7             	mov    %rsp,%rdi
  401413:	e8 08 f8 ff ff       	callq  400c20 <gethostname@plt>
  401418:	85 c0                	test   %eax,%eax
  40141a:	74 14                	je     401430 <initialize_bomb+0x4e>
  40141c:	bf d8 26 40 00       	mov    $0x4026d8,%edi
  401421:	e8 fa f6 ff ff       	callq  400b20 <puts@plt>
  401426:	bf 08 00 00 00       	mov    $0x8,%edi
  40142b:	e8 00 f8 ff ff       	callq  400c30 <exit@plt>
  401430:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  401435:	e8 59 0d 00 00       	callq  402193 <init_driver>
  40143a:	85 c0                	test   %eax,%eax
  40143c:	79 23                	jns    401461 <initialize_bomb+0x7f>
  40143e:	48 8d 54 24 40       	lea    0x40(%rsp),%rdx
  401443:	be 52 28 40 00       	mov    $0x402852,%esi
  401448:	bf 01 00 00 00       	mov    $0x1,%edi
  40144d:	b8 00 00 00 00       	mov    $0x0,%eax
  401452:	e8 a9 f7 ff ff       	callq  400c00 <__printf_chk@plt>
  401457:	bf 08 00 00 00       	mov    $0x8,%edi
  40145c:	e8 cf f7 ff ff       	callq  400c30 <exit@plt>
  401461:	48 8b 84 24 48 20 00 	mov    0x2048(%rsp),%rax
  401468:	00 
  401469:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  401470:	00 00 
  401472:	74 05                	je     401479 <initialize_bomb+0x97>
  401474:	e8 c7 f6 ff ff       	callq  400b40 <__stack_chk_fail@plt>
  401479:	48 81 c4 58 20 00 00 	add    $0x2058,%rsp
  401480:	c3                   	retq   

0000000000401481 <initialize_bomb_solve>:
  401481:	f3 c3                	repz retq 

0000000000401483 <blank_line>:
  401483:	55                   	push   %rbp
  401484:	53                   	push   %rbx
  401485:	48 83 ec 08          	sub    $0x8,%rsp
  401489:	48 89 fd             	mov    %rdi,%rbp
  40148c:	eb 17                	jmp    4014a5 <blank_line+0x22>
  40148e:	e8 dd f7 ff ff       	callq  400c70 <__ctype_b_loc@plt>
  401493:	48 83 c5 01          	add    $0x1,%rbp
  401497:	48 0f be db          	movsbq %bl,%rbx
  40149b:	48 8b 00             	mov    (%rax),%rax
  40149e:	f6 44 58 01 20       	testb  $0x20,0x1(%rax,%rbx,2)
  4014a3:	74 0f                	je     4014b4 <blank_line+0x31>
  4014a5:	0f b6 5d 00          	movzbl 0x0(%rbp),%ebx
  4014a9:	84 db                	test   %bl,%bl
  4014ab:	75 e1                	jne    40148e <blank_line+0xb>
  4014ad:	b8 01 00 00 00       	mov    $0x1,%eax
  4014b2:	eb 05                	jmp    4014b9 <blank_line+0x36>
  4014b4:	b8 00 00 00 00       	mov    $0x0,%eax
  4014b9:	48 83 c4 08          	add    $0x8,%rsp
  4014bd:	5b                   	pop    %rbx
  4014be:	5d                   	pop    %rbp
  4014bf:	c3                   	retq   

00000000004014c0 <skip>:
  4014c0:	53                   	push   %rbx
  4014c1:	48 63 05 e4 36 20 00 	movslq 0x2036e4(%rip),%rax        # 604bac <num_input_strings>
  4014c8:	48 8d 3c 80          	lea    (%rax,%rax,4),%rdi
  4014cc:	48 c1 e7 04          	shl    $0x4,%rdi
  4014d0:	48 81 c7 c0 4b 60 00 	add    $0x604bc0,%rdi
  4014d7:	48 8b 15 d2 36 20 00 	mov    0x2036d2(%rip),%rdx        # 604bb0 <infile>
  4014de:	be 50 00 00 00       	mov    $0x50,%esi
  4014e3:	e8 a8 f6 ff ff       	callq  400b90 <fgets@plt>
  4014e8:	48 89 c3             	mov    %rax,%rbx
  4014eb:	48 85 c0             	test   %rax,%rax
  4014ee:	74 0c                	je     4014fc <skip+0x3c>
  4014f0:	48 89 c7             	mov    %rax,%rdi
  4014f3:	e8 8b ff ff ff       	callq  401483 <blank_line>
  4014f8:	85 c0                	test   %eax,%eax
  4014fa:	75 c5                	jne    4014c1 <skip+0x1>
  4014fc:	48 89 d8             	mov    %rbx,%rax
  4014ff:	5b                   	pop    %rbx
  401500:	c3                   	retq   

0000000000401501 <send_msg>:
  401501:	48 81 ec 18 40 00 00 	sub    $0x4018,%rsp
  401508:	41 89 f8             	mov    %edi,%r8d
  40150b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401512:	00 00 
  401514:	48 89 84 24 08 40 00 	mov    %rax,0x4008(%rsp)
  40151b:	00 
  40151c:	31 c0                	xor    %eax,%eax
  40151e:	8b 35 88 36 20 00    	mov    0x203688(%rip),%esi        # 604bac <num_input_strings>
  401524:	8d 46 ff             	lea    -0x1(%rsi),%eax
  401527:	48 98                	cltq   
  401529:	48 8d 14 80          	lea    (%rax,%rax,4),%rdx
  40152d:	48 c1 e2 04          	shl    $0x4,%rdx
  401531:	48 81 c2 c0 4b 60 00 	add    $0x604bc0,%rdx
  401538:	b8 00 00 00 00       	mov    $0x0,%eax
  40153d:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
  401544:	48 89 d7             	mov    %rdx,%rdi
  401547:	f2 ae                	repnz scas %es:(%rdi),%al
  401549:	48 f7 d1             	not    %rcx
  40154c:	48 83 c1 63          	add    $0x63,%rcx
  401550:	48 81 f9 00 20 00 00 	cmp    $0x2000,%rcx
  401557:	76 19                	jbe    401572 <send_msg+0x71>
  401559:	be 10 27 40 00       	mov    $0x402710,%esi
  40155e:	bf 01 00 00 00       	mov    $0x1,%edi
  401563:	e8 98 f6 ff ff       	callq  400c00 <__printf_chk@plt>
  401568:	bf 08 00 00 00       	mov    $0x8,%edi
  40156d:	e8 be f6 ff ff       	callq  400c30 <exit@plt>
  401572:	45 85 c0             	test   %r8d,%r8d
  401575:	41 b9 74 28 40 00    	mov    $0x402874,%r9d
  40157b:	b8 6c 28 40 00       	mov    $0x40286c,%eax
  401580:	4c 0f 45 c8          	cmovne %rax,%r9
  401584:	52                   	push   %rdx
  401585:	56                   	push   %rsi
  401586:	44 8b 05 d3 31 20 00 	mov    0x2031d3(%rip),%r8d        # 604760 <bomb_id>
  40158d:	b9 7d 28 40 00       	mov    $0x40287d,%ecx
  401592:	ba 00 20 00 00       	mov    $0x2000,%edx
  401597:	be 01 00 00 00       	mov    $0x1,%esi
  40159c:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  4015a1:	b8 00 00 00 00       	mov    $0x0,%eax
  4015a6:	e8 d5 f6 ff ff       	callq  400c80 <__sprintf_chk@plt>
  4015ab:	48 8d 8c 24 10 20 00 	lea    0x2010(%rsp),%rcx
  4015b2:	00 
  4015b3:	ba 00 00 00 00       	mov    $0x0,%edx
  4015b8:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
  4015bd:	bf 60 43 60 00       	mov    $0x604360,%edi
  4015c2:	e8 a1 0d 00 00       	callq  402368 <driver_post>
  4015c7:	48 83 c4 10          	add    $0x10,%rsp
  4015cb:	85 c0                	test   %eax,%eax
  4015cd:	79 17                	jns    4015e6 <send_msg+0xe5>
  4015cf:	48 8d bc 24 00 20 00 	lea    0x2000(%rsp),%rdi
  4015d6:	00 
  4015d7:	e8 44 f5 ff ff       	callq  400b20 <puts@plt>
  4015dc:	bf 00 00 00 00       	mov    $0x0,%edi
  4015e1:	e8 4a f6 ff ff       	callq  400c30 <exit@plt>
  4015e6:	48 8b 84 24 08 40 00 	mov    0x4008(%rsp),%rax
  4015ed:	00 
  4015ee:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4015f5:	00 00 
  4015f7:	74 05                	je     4015fe <send_msg+0xfd>
  4015f9:	e8 42 f5 ff ff       	callq  400b40 <__stack_chk_fail@plt>
  4015fe:	48 81 c4 18 40 00 00 	add    $0x4018,%rsp
  401605:	c3                   	retq   

0000000000401606 <explode_bomb>:
  401606:	48 83 ec 08          	sub    $0x8,%rsp
  40160a:	bf 89 28 40 00       	mov    $0x402889,%edi
  40160f:	e8 0c f5 ff ff       	callq  400b20 <puts@plt>
  401614:	bf 92 28 40 00       	mov    $0x402892,%edi
  401619:	e8 02 f5 ff ff       	callq  400b20 <puts@plt>
  40161e:	bf 00 00 00 00       	mov    $0x0,%edi
  401623:	e8 d9 fe ff ff       	callq  401501 <send_msg>
  401628:	bf 38 27 40 00       	mov    $0x402738,%edi
  40162d:	e8 ee f4 ff ff       	callq  400b20 <puts@plt>
  401632:	bf 08 00 00 00       	mov    $0x8,%edi
  401637:	e8 f4 f5 ff ff       	callq  400c30 <exit@plt>

000000000040163c <read_six_numbers>:
  40163c:	48 83 ec 08          	sub    $0x8,%rsp
  401640:	48 89 f2             	mov    %rsi,%rdx
  401643:	48 8d 4e 04          	lea    0x4(%rsi),%rcx
  401647:	48 8d 46 14          	lea    0x14(%rsi),%rax
  40164b:	50                   	push   %rax
  40164c:	48 8d 46 10          	lea    0x10(%rsi),%rax
  401650:	50                   	push   %rax
  401651:	4c 8d 4e 0c          	lea    0xc(%rsi),%r9
  401655:	4c 8d 46 08          	lea    0x8(%rsi),%r8
  401659:	be a9 28 40 00       	mov    $0x4028a9,%esi
  40165e:	b8 00 00 00 00       	mov    $0x0,%eax
  401663:	e8 88 f5 ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  401668:	48 83 c4 10          	add    $0x10,%rsp
  40166c:	83 f8 05             	cmp    $0x5,%eax
  40166f:	7f 05                	jg     401676 <read_six_numbers+0x3a>
  401671:	e8 90 ff ff ff       	callq  401606 <explode_bomb>
  401676:	48 83 c4 08          	add    $0x8,%rsp
  40167a:	c3                   	retq   

000000000040167b <read_line>:
  40167b:	48 83 ec 08          	sub    $0x8,%rsp
  40167f:	b8 00 00 00 00       	mov    $0x0,%eax
  401684:	e8 37 fe ff ff       	callq  4014c0 <skip>
  401689:	48 85 c0             	test   %rax,%rax
  40168c:	75 6e                	jne    4016fc <read_line+0x81>
  40168e:	48 8b 05 fb 34 20 00 	mov    0x2034fb(%rip),%rax        # 604b90 <stdin@@GLIBC_2.2.5>
  401695:	48 39 05 14 35 20 00 	cmp    %rax,0x203514(%rip)        # 604bb0 <infile>
  40169c:	75 14                	jne    4016b2 <read_line+0x37>
  40169e:	bf bb 28 40 00       	mov    $0x4028bb,%edi
  4016a3:	e8 78 f4 ff ff       	callq  400b20 <puts@plt>
  4016a8:	bf 08 00 00 00       	mov    $0x8,%edi
  4016ad:	e8 7e f5 ff ff       	callq  400c30 <exit@plt>
  4016b2:	bf d9 28 40 00       	mov    $0x4028d9,%edi
  4016b7:	e8 34 f4 ff ff       	callq  400af0 <getenv@plt>
  4016bc:	48 85 c0             	test   %rax,%rax
  4016bf:	74 0a                	je     4016cb <read_line+0x50>
  4016c1:	bf 00 00 00 00       	mov    $0x0,%edi
  4016c6:	e8 65 f5 ff ff       	callq  400c30 <exit@plt>
  4016cb:	48 8b 05 be 34 20 00 	mov    0x2034be(%rip),%rax        # 604b90 <stdin@@GLIBC_2.2.5>
  4016d2:	48 89 05 d7 34 20 00 	mov    %rax,0x2034d7(%rip)        # 604bb0 <infile>
  4016d9:	b8 00 00 00 00       	mov    $0x0,%eax
  4016de:	e8 dd fd ff ff       	callq  4014c0 <skip>
  4016e3:	48 85 c0             	test   %rax,%rax
  4016e6:	75 14                	jne    4016fc <read_line+0x81>
  4016e8:	bf bb 28 40 00       	mov    $0x4028bb,%edi
  4016ed:	e8 2e f4 ff ff       	callq  400b20 <puts@plt>
  4016f2:	bf 00 00 00 00       	mov    $0x0,%edi
  4016f7:	e8 34 f5 ff ff       	callq  400c30 <exit@plt>
  4016fc:	8b 35 aa 34 20 00    	mov    0x2034aa(%rip),%esi        # 604bac <num_input_strings>
  401702:	48 63 c6             	movslq %esi,%rax
  401705:	48 8d 14 80          	lea    (%rax,%rax,4),%rdx
  401709:	48 c1 e2 04          	shl    $0x4,%rdx
  40170d:	48 81 c2 c0 4b 60 00 	add    $0x604bc0,%rdx
  401714:	b8 00 00 00 00       	mov    $0x0,%eax
  401719:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
  401720:	48 89 d7             	mov    %rdx,%rdi
  401723:	f2 ae                	repnz scas %es:(%rdi),%al
  401725:	48 f7 d1             	not    %rcx
  401728:	48 83 e9 01          	sub    $0x1,%rcx
  40172c:	83 f9 4e             	cmp    $0x4e,%ecx
  40172f:	7e 46                	jle    401777 <read_line+0xfc>
  401731:	bf e4 28 40 00       	mov    $0x4028e4,%edi
  401736:	e8 e5 f3 ff ff       	callq  400b20 <puts@plt>
  40173b:	8b 05 6b 34 20 00    	mov    0x20346b(%rip),%eax        # 604bac <num_input_strings>
  401741:	8d 50 01             	lea    0x1(%rax),%edx
  401744:	89 15 62 34 20 00    	mov    %edx,0x203462(%rip)        # 604bac <num_input_strings>
  40174a:	48 98                	cltq   
  40174c:	48 6b c0 50          	imul   $0x50,%rax,%rax
  401750:	48 bf 2a 2a 2a 74 72 	movabs $0x636e7572742a2a2a,%rdi
  401757:	75 6e 63 
  40175a:	48 89 b8 c0 4b 60 00 	mov    %rdi,0x604bc0(%rax)
  401761:	48 bf 61 74 65 64 2a 	movabs $0x2a2a2a64657461,%rdi
  401768:	2a 2a 00 
  40176b:	48 89 b8 c8 4b 60 00 	mov    %rdi,0x604bc8(%rax)
  401772:	e8 8f fe ff ff       	callq  401606 <explode_bomb>
  401777:	83 e9 01             	sub    $0x1,%ecx
  40177a:	48 63 c9             	movslq %ecx,%rcx
  40177d:	48 63 c6             	movslq %esi,%rax
  401780:	48 8d 04 80          	lea    (%rax,%rax,4),%rax
  401784:	48 c1 e0 04          	shl    $0x4,%rax
  401788:	c6 84 01 c0 4b 60 00 	movb   $0x0,0x604bc0(%rcx,%rax,1)
  40178f:	00 
  401790:	8d 46 01             	lea    0x1(%rsi),%eax
  401793:	89 05 13 34 20 00    	mov    %eax,0x203413(%rip)        # 604bac <num_input_strings>
  401799:	48 89 d0             	mov    %rdx,%rax
  40179c:	48 83 c4 08          	add    $0x8,%rsp
  4017a0:	c3                   	retq   

00000000004017a1 <phase_defused>:
  4017a1:	48 83 ec 78          	sub    $0x78,%rsp
  4017a5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4017ac:	00 00 
  4017ae:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  4017b3:	31 c0                	xor    %eax,%eax
  4017b5:	bf 01 00 00 00       	mov    $0x1,%edi
  4017ba:	e8 42 fd ff ff       	callq  401501 <send_msg>
  4017bf:	83 3d e6 33 20 00 06 	cmpl   $0x6,0x2033e6(%rip)        # 604bac <num_input_strings>
  4017c6:	75 6d                	jne    401835 <phase_defused+0x94>
  4017c8:	4c 8d 44 24 10       	lea    0x10(%rsp),%r8
  4017cd:	48 8d 4c 24 0c       	lea    0xc(%rsp),%rcx
  4017d2:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
  4017d7:	be ff 28 40 00       	mov    $0x4028ff,%esi
  4017dc:	bf b0 4c 60 00       	mov    $0x604cb0,%edi
  4017e1:	b8 00 00 00 00       	mov    $0x0,%eax
  4017e6:	e8 05 f4 ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  4017eb:	83 f8 03             	cmp    $0x3,%eax
  4017ee:	75 31                	jne    401821 <phase_defused+0x80>
  4017f0:	be 08 29 40 00       	mov    $0x402908,%esi
  4017f5:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  4017fa:	e8 7c fb ff ff       	callq  40137b <strings_not_equal>
  4017ff:	85 c0                	test   %eax,%eax
  401801:	75 1e                	jne    401821 <phase_defused+0x80>
  401803:	bf 60 27 40 00       	mov    $0x402760,%edi
  401808:	e8 13 f3 ff ff       	callq  400b20 <puts@plt>
  40180d:	bf 88 27 40 00       	mov    $0x402788,%edi
  401812:	e8 09 f3 ff ff       	callq  400b20 <puts@plt>
  401817:	b8 00 00 00 00       	mov    $0x0,%eax
  40181c:	e8 70 fa ff ff       	callq  401291 <secret_phase>
  401821:	bf c0 27 40 00       	mov    $0x4027c0,%edi
  401826:	e8 f5 f2 ff ff       	callq  400b20 <puts@plt>
  40182b:	bf f0 27 40 00       	mov    $0x4027f0,%edi
  401830:	e8 eb f2 ff ff       	callq  400b20 <puts@plt>
  401835:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  40183a:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  401841:	00 00 
  401843:	74 05                	je     40184a <phase_defused+0xa9>
  401845:	e8 f6 f2 ff ff       	callq  400b40 <__stack_chk_fail@plt>
  40184a:	48 83 c4 78          	add    $0x78,%rsp
  40184e:	c3                   	retq   

000000000040184f <sigalrm_handler>:
  40184f:	48 83 ec 08          	sub    $0x8,%rsp
  401853:	b9 00 00 00 00       	mov    $0x0,%ecx
  401858:	ba 60 29 40 00       	mov    $0x402960,%edx
  40185d:	be 01 00 00 00       	mov    $0x1,%esi
  401862:	48 8b 3d 37 33 20 00 	mov    0x203337(%rip),%rdi        # 604ba0 <stderr@@GLIBC_2.2.5>
  401869:	b8 00 00 00 00       	mov    $0x0,%eax
  40186e:	e8 dd f3 ff ff       	callq  400c50 <__fprintf_chk@plt>
  401873:	bf 01 00 00 00       	mov    $0x1,%edi
  401878:	e8 b3 f3 ff ff       	callq  400c30 <exit@plt>

000000000040187d <rio_readlineb>:
  40187d:	41 56                	push   %r14
  40187f:	41 55                	push   %r13
  401881:	41 54                	push   %r12
  401883:	55                   	push   %rbp
  401884:	53                   	push   %rbx
  401885:	48 83 ec 10          	sub    $0x10,%rsp
  401889:	48 89 fb             	mov    %rdi,%rbx
  40188c:	49 89 f5             	mov    %rsi,%r13
  40188f:	49 89 d6             	mov    %rdx,%r14
  401892:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401899:	00 00 
  40189b:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4018a0:	31 c0                	xor    %eax,%eax
  4018a2:	41 bc 01 00 00 00    	mov    $0x1,%r12d
  4018a8:	48 8d 6f 10          	lea    0x10(%rdi),%rbp
  4018ac:	48 83 fa 01          	cmp    $0x1,%rdx
  4018b0:	77 2c                	ja     4018de <rio_readlineb+0x61>
  4018b2:	eb 6d                	jmp    401921 <rio_readlineb+0xa4>
  4018b4:	ba 00 20 00 00       	mov    $0x2000,%edx
  4018b9:	48 89 ee             	mov    %rbp,%rsi
  4018bc:	8b 3b                	mov    (%rbx),%edi
  4018be:	e8 ad f2 ff ff       	callq  400b70 <read@plt>
  4018c3:	89 43 04             	mov    %eax,0x4(%rbx)
  4018c6:	85 c0                	test   %eax,%eax
  4018c8:	79 0c                	jns    4018d6 <rio_readlineb+0x59>
  4018ca:	e8 31 f2 ff ff       	callq  400b00 <__errno_location@plt>
  4018cf:	83 38 04             	cmpl   $0x4,(%rax)
  4018d2:	74 0a                	je     4018de <rio_readlineb+0x61>
  4018d4:	eb 6c                	jmp    401942 <rio_readlineb+0xc5>
  4018d6:	85 c0                	test   %eax,%eax
  4018d8:	74 71                	je     40194b <rio_readlineb+0xce>
  4018da:	48 89 6b 08          	mov    %rbp,0x8(%rbx)
  4018de:	8b 43 04             	mov    0x4(%rbx),%eax
  4018e1:	85 c0                	test   %eax,%eax
  4018e3:	7e cf                	jle    4018b4 <rio_readlineb+0x37>
  4018e5:	48 8b 53 08          	mov    0x8(%rbx),%rdx
  4018e9:	0f b6 0a             	movzbl (%rdx),%ecx
  4018ec:	88 4c 24 07          	mov    %cl,0x7(%rsp)
  4018f0:	48 83 c2 01          	add    $0x1,%rdx
  4018f4:	48 89 53 08          	mov    %rdx,0x8(%rbx)
  4018f8:	83 e8 01             	sub    $0x1,%eax
  4018fb:	89 43 04             	mov    %eax,0x4(%rbx)
  4018fe:	49 83 c5 01          	add    $0x1,%r13
  401902:	41 88 4d ff          	mov    %cl,-0x1(%r13)
  401906:	80 f9 0a             	cmp    $0xa,%cl
  401909:	75 0a                	jne    401915 <rio_readlineb+0x98>
  40190b:	eb 14                	jmp    401921 <rio_readlineb+0xa4>
  40190d:	41 83 fc 01          	cmp    $0x1,%r12d
  401911:	75 0e                	jne    401921 <rio_readlineb+0xa4>
  401913:	eb 16                	jmp    40192b <rio_readlineb+0xae>
  401915:	41 83 c4 01          	add    $0x1,%r12d
  401919:	49 63 c4             	movslq %r12d,%rax
  40191c:	4c 39 f0             	cmp    %r14,%rax
  40191f:	72 bd                	jb     4018de <rio_readlineb+0x61>
  401921:	41 c6 45 00 00       	movb   $0x0,0x0(%r13)
  401926:	49 63 c4             	movslq %r12d,%rax
  401929:	eb 05                	jmp    401930 <rio_readlineb+0xb3>
  40192b:	b8 00 00 00 00       	mov    $0x0,%eax
  401930:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
  401935:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
  40193c:	00 00 
  40193e:	74 22                	je     401962 <rio_readlineb+0xe5>
  401940:	eb 1b                	jmp    40195d <rio_readlineb+0xe0>
  401942:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  401949:	eb 05                	jmp    401950 <rio_readlineb+0xd3>
  40194b:	b8 00 00 00 00       	mov    $0x0,%eax
  401950:	85 c0                	test   %eax,%eax
  401952:	74 b9                	je     40190d <rio_readlineb+0x90>
  401954:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  40195b:	eb d3                	jmp    401930 <rio_readlineb+0xb3>
  40195d:	e8 de f1 ff ff       	callq  400b40 <__stack_chk_fail@plt>
  401962:	48 83 c4 10          	add    $0x10,%rsp
  401966:	5b                   	pop    %rbx
  401967:	5d                   	pop    %rbp
  401968:	41 5c                	pop    %r12
  40196a:	41 5d                	pop    %r13
  40196c:	41 5e                	pop    %r14
  40196e:	c3                   	retq   

000000000040196f <submitr>:
  40196f:	41 57                	push   %r15
  401971:	41 56                	push   %r14
  401973:	41 55                	push   %r13
  401975:	41 54                	push   %r12
  401977:	55                   	push   %rbp
  401978:	53                   	push   %rbx
  401979:	48 81 ec 68 a0 00 00 	sub    $0xa068,%rsp
  401980:	48 89 fd             	mov    %rdi,%rbp
  401983:	41 89 f5             	mov    %esi,%r13d
  401986:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
  40198b:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  401990:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
  401995:	4c 89 cb             	mov    %r9,%rbx
  401998:	4c 8b bc 24 a0 a0 00 	mov    0xa0a0(%rsp),%r15
  40199f:	00 
  4019a0:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4019a7:	00 00 
  4019a9:	48 89 84 24 58 a0 00 	mov    %rax,0xa058(%rsp)
  4019b0:	00 
  4019b1:	31 c0                	xor    %eax,%eax
  4019b3:	c7 44 24 2c 00 00 00 	movl   $0x0,0x2c(%rsp)
  4019ba:	00 
  4019bb:	ba 00 00 00 00       	mov    $0x0,%edx
  4019c0:	be 01 00 00 00       	mov    $0x1,%esi
  4019c5:	bf 02 00 00 00       	mov    $0x2,%edi
  4019ca:	e8 c1 f2 ff ff       	callq  400c90 <socket@plt>
  4019cf:	85 c0                	test   %eax,%eax
  4019d1:	79 50                	jns    401a23 <submitr+0xb4>
  4019d3:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  4019da:	3a 20 43 
  4019dd:	49 89 07             	mov    %rax,(%r15)
  4019e0:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  4019e7:	20 75 6e 
  4019ea:	49 89 47 08          	mov    %rax,0x8(%r15)
  4019ee:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  4019f5:	74 6f 20 
  4019f8:	49 89 47 10          	mov    %rax,0x10(%r15)
  4019fc:	48 b8 63 72 65 61 74 	movabs $0x7320657461657263,%rax
  401a03:	65 20 73 
  401a06:	49 89 47 18          	mov    %rax,0x18(%r15)
  401a0a:	41 c7 47 20 6f 63 6b 	movl   $0x656b636f,0x20(%r15)
  401a11:	65 
  401a12:	66 41 c7 47 24 74 00 	movw   $0x74,0x24(%r15)
  401a19:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  401a1e:	e9 15 06 00 00       	jmpq   402038 <submitr+0x6c9>
  401a23:	41 89 c4             	mov    %eax,%r12d
  401a26:	48 89 ef             	mov    %rbp,%rdi
  401a29:	e8 82 f1 ff ff       	callq  400bb0 <gethostbyname@plt>
  401a2e:	48 85 c0             	test   %rax,%rax
  401a31:	75 6b                	jne    401a9e <submitr+0x12f>
  401a33:	48 b8 45 72 72 6f 72 	movabs $0x44203a726f727245,%rax
  401a3a:	3a 20 44 
  401a3d:	49 89 07             	mov    %rax,(%r15)
  401a40:	48 b8 4e 53 20 69 73 	movabs $0x6e7520736920534e,%rax
  401a47:	20 75 6e 
  401a4a:	49 89 47 08          	mov    %rax,0x8(%r15)
  401a4e:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  401a55:	74 6f 20 
  401a58:	49 89 47 10          	mov    %rax,0x10(%r15)
  401a5c:	48 b8 72 65 73 6f 6c 	movabs $0x2065766c6f736572,%rax
  401a63:	76 65 20 
  401a66:	49 89 47 18          	mov    %rax,0x18(%r15)
  401a6a:	48 b8 73 65 72 76 65 	movabs $0x6120726576726573,%rax
  401a71:	72 20 61 
  401a74:	49 89 47 20          	mov    %rax,0x20(%r15)
  401a78:	41 c7 47 28 64 64 72 	movl   $0x65726464,0x28(%r15)
  401a7f:	65 
  401a80:	66 41 c7 47 2c 73 73 	movw   $0x7373,0x2c(%r15)
  401a87:	41 c6 47 2e 00       	movb   $0x0,0x2e(%r15)
  401a8c:	44 89 e7             	mov    %r12d,%edi
  401a8f:	e8 cc f0 ff ff       	callq  400b60 <close@plt>
  401a94:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  401a99:	e9 9a 05 00 00       	jmpq   402038 <submitr+0x6c9>
  401a9e:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
  401aa5:	00 00 
  401aa7:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
  401aae:	00 00 
  401ab0:	66 c7 44 24 30 02 00 	movw   $0x2,0x30(%rsp)
  401ab7:	48 63 50 14          	movslq 0x14(%rax),%rdx
  401abb:	48 8b 40 18          	mov    0x18(%rax),%rax
  401abf:	48 8d 7c 24 34       	lea    0x34(%rsp),%rdi
  401ac4:	b9 0c 00 00 00       	mov    $0xc,%ecx
  401ac9:	48 8b 30             	mov    (%rax),%rsi
  401acc:	e8 ef f0 ff ff       	callq  400bc0 <__memmove_chk@plt>
  401ad1:	66 41 c1 cd 08       	ror    $0x8,%r13w
  401ad6:	66 44 89 6c 24 32    	mov    %r13w,0x32(%rsp)
  401adc:	ba 10 00 00 00       	mov    $0x10,%edx
  401ae1:	48 8d 74 24 30       	lea    0x30(%rsp),%rsi
  401ae6:	44 89 e7             	mov    %r12d,%edi
  401ae9:	e8 52 f1 ff ff       	callq  400c40 <connect@plt>
  401aee:	85 c0                	test   %eax,%eax
  401af0:	79 5d                	jns    401b4f <submitr+0x1e0>
  401af2:	48 b8 45 72 72 6f 72 	movabs $0x55203a726f727245,%rax
  401af9:	3a 20 55 
  401afc:	49 89 07             	mov    %rax,(%r15)
  401aff:	48 b8 6e 61 62 6c 65 	movabs $0x6f7420656c62616e,%rax
  401b06:	20 74 6f 
  401b09:	49 89 47 08          	mov    %rax,0x8(%r15)
  401b0d:	48 b8 20 63 6f 6e 6e 	movabs $0x7463656e6e6f6320,%rax
  401b14:	65 63 74 
  401b17:	49 89 47 10          	mov    %rax,0x10(%r15)
  401b1b:	48 b8 20 74 6f 20 74 	movabs $0x20656874206f7420,%rax
  401b22:	68 65 20 
  401b25:	49 89 47 18          	mov    %rax,0x18(%r15)
  401b29:	41 c7 47 20 73 65 72 	movl   $0x76726573,0x20(%r15)
  401b30:	76 
  401b31:	66 41 c7 47 24 65 72 	movw   $0x7265,0x24(%r15)
  401b38:	41 c6 47 26 00       	movb   $0x0,0x26(%r15)
  401b3d:	44 89 e7             	mov    %r12d,%edi
  401b40:	e8 1b f0 ff ff       	callq  400b60 <close@plt>
  401b45:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  401b4a:	e9 e9 04 00 00       	jmpq   402038 <submitr+0x6c9>
  401b4f:	49 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%r9
  401b56:	b8 00 00 00 00       	mov    $0x0,%eax
  401b5b:	4c 89 c9             	mov    %r9,%rcx
  401b5e:	48 89 df             	mov    %rbx,%rdi
  401b61:	f2 ae                	repnz scas %es:(%rdi),%al
  401b63:	48 f7 d1             	not    %rcx
  401b66:	48 89 ce             	mov    %rcx,%rsi
  401b69:	4c 89 c9             	mov    %r9,%rcx
  401b6c:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  401b71:	f2 ae                	repnz scas %es:(%rdi),%al
  401b73:	49 89 c8             	mov    %rcx,%r8
  401b76:	4c 89 c9             	mov    %r9,%rcx
  401b79:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  401b7e:	f2 ae                	repnz scas %es:(%rdi),%al
  401b80:	48 f7 d1             	not    %rcx
  401b83:	48 89 ca             	mov    %rcx,%rdx
  401b86:	4c 89 c9             	mov    %r9,%rcx
  401b89:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  401b8e:	f2 ae                	repnz scas %es:(%rdi),%al
  401b90:	4c 29 c2             	sub    %r8,%rdx
  401b93:	48 29 ca             	sub    %rcx,%rdx
  401b96:	48 8d 44 76 fd       	lea    -0x3(%rsi,%rsi,2),%rax
  401b9b:	48 8d 44 02 7b       	lea    0x7b(%rdx,%rax,1),%rax
  401ba0:	48 3d 00 20 00 00    	cmp    $0x2000,%rax
  401ba6:	76 73                	jbe    401c1b <submitr+0x2ac>
  401ba8:	48 b8 45 72 72 6f 72 	movabs $0x52203a726f727245,%rax
  401baf:	3a 20 52 
  401bb2:	49 89 07             	mov    %rax,(%r15)
  401bb5:	48 b8 65 73 75 6c 74 	movabs $0x747320746c757365,%rax
  401bbc:	20 73 74 
  401bbf:	49 89 47 08          	mov    %rax,0x8(%r15)
  401bc3:	48 b8 72 69 6e 67 20 	movabs $0x6f6f7420676e6972,%rax
  401bca:	74 6f 6f 
  401bcd:	49 89 47 10          	mov    %rax,0x10(%r15)
  401bd1:	48 b8 20 6c 61 72 67 	movabs $0x202e656772616c20,%rax
  401bd8:	65 2e 20 
  401bdb:	49 89 47 18          	mov    %rax,0x18(%r15)
  401bdf:	48 b8 49 6e 63 72 65 	movabs $0x6573616572636e49,%rax
  401be6:	61 73 65 
  401be9:	49 89 47 20          	mov    %rax,0x20(%r15)
  401bed:	48 b8 20 53 55 42 4d 	movabs $0x5254494d42555320,%rax
  401bf4:	49 54 52 
  401bf7:	49 89 47 28          	mov    %rax,0x28(%r15)
  401bfb:	48 b8 5f 4d 41 58 42 	movabs $0x46554258414d5f,%rax
  401c02:	55 46 00 
  401c05:	49 89 47 30          	mov    %rax,0x30(%r15)
  401c09:	44 89 e7             	mov    %r12d,%edi
  401c0c:	e8 4f ef ff ff       	callq  400b60 <close@plt>
  401c11:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  401c16:	e9 1d 04 00 00       	jmpq   402038 <submitr+0x6c9>
  401c1b:	48 8d 94 24 50 40 00 	lea    0x4050(%rsp),%rdx
  401c22:	00 
  401c23:	b9 00 04 00 00       	mov    $0x400,%ecx
  401c28:	b8 00 00 00 00       	mov    $0x0,%eax
  401c2d:	48 89 d7             	mov    %rdx,%rdi
  401c30:	f3 48 ab             	rep stos %rax,%es:(%rdi)
  401c33:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
  401c3a:	48 89 df             	mov    %rbx,%rdi
  401c3d:	f2 ae                	repnz scas %es:(%rdi),%al
  401c3f:	48 89 c8             	mov    %rcx,%rax
  401c42:	48 f7 d0             	not    %rax
  401c45:	48 83 e8 01          	sub    $0x1,%rax
  401c49:	85 c0                	test   %eax,%eax
  401c4b:	0f 84 90 04 00 00    	je     4020e1 <submitr+0x772>
  401c51:	8d 40 ff             	lea    -0x1(%rax),%eax
  401c54:	4c 8d 74 03 01       	lea    0x1(%rbx,%rax,1),%r14
  401c59:	48 89 d5             	mov    %rdx,%rbp
  401c5c:	49 bd d9 ff 00 00 00 	movabs $0x2000000000ffd9,%r13
  401c63:	00 20 00 
  401c66:	44 0f b6 03          	movzbl (%rbx),%r8d
  401c6a:	41 8d 40 d6          	lea    -0x2a(%r8),%eax
  401c6e:	3c 35                	cmp    $0x35,%al
  401c70:	77 06                	ja     401c78 <submitr+0x309>
  401c72:	49 0f a3 c5          	bt     %rax,%r13
  401c76:	72 0d                	jb     401c85 <submitr+0x316>
  401c78:	44 89 c0             	mov    %r8d,%eax
  401c7b:	83 e0 df             	and    $0xffffffdf,%eax
  401c7e:	83 e8 41             	sub    $0x41,%eax
  401c81:	3c 19                	cmp    $0x19,%al
  401c83:	77 0a                	ja     401c8f <submitr+0x320>
  401c85:	44 88 45 00          	mov    %r8b,0x0(%rbp)
  401c89:	48 8d 6d 01          	lea    0x1(%rbp),%rbp
  401c8d:	eb 6c                	jmp    401cfb <submitr+0x38c>
  401c8f:	41 80 f8 20          	cmp    $0x20,%r8b
  401c93:	75 0a                	jne    401c9f <submitr+0x330>
  401c95:	c6 45 00 2b          	movb   $0x2b,0x0(%rbp)
  401c99:	48 8d 6d 01          	lea    0x1(%rbp),%rbp
  401c9d:	eb 5c                	jmp    401cfb <submitr+0x38c>
  401c9f:	41 8d 40 e0          	lea    -0x20(%r8),%eax
  401ca3:	3c 5f                	cmp    $0x5f,%al
  401ca5:	76 0a                	jbe    401cb1 <submitr+0x342>
  401ca7:	41 80 f8 09          	cmp    $0x9,%r8b
  401cab:	0f 85 a3 03 00 00    	jne    402054 <submitr+0x6e5>
  401cb1:	45 0f b6 c0          	movzbl %r8b,%r8d
  401cb5:	b9 30 2a 40 00       	mov    $0x402a30,%ecx
  401cba:	ba 08 00 00 00       	mov    $0x8,%edx
  401cbf:	be 01 00 00 00       	mov    $0x1,%esi
  401cc4:	48 8d bc 24 50 80 00 	lea    0x8050(%rsp),%rdi
  401ccb:	00 
  401ccc:	b8 00 00 00 00       	mov    $0x0,%eax
  401cd1:	e8 aa ef ff ff       	callq  400c80 <__sprintf_chk@plt>
  401cd6:	0f b6 84 24 50 80 00 	movzbl 0x8050(%rsp),%eax
  401cdd:	00 
  401cde:	88 45 00             	mov    %al,0x0(%rbp)
  401ce1:	0f b6 84 24 51 80 00 	movzbl 0x8051(%rsp),%eax
  401ce8:	00 
  401ce9:	88 45 01             	mov    %al,0x1(%rbp)
  401cec:	0f b6 84 24 52 80 00 	movzbl 0x8052(%rsp),%eax
  401cf3:	00 
  401cf4:	88 45 02             	mov    %al,0x2(%rbp)
  401cf7:	48 8d 6d 03          	lea    0x3(%rbp),%rbp
  401cfb:	48 83 c3 01          	add    $0x1,%rbx
  401cff:	49 39 de             	cmp    %rbx,%r14
  401d02:	0f 85 5e ff ff ff    	jne    401c66 <submitr+0x2f7>
  401d08:	e9 d4 03 00 00       	jmpq   4020e1 <submitr+0x772>
  401d0d:	48 89 da             	mov    %rbx,%rdx
  401d10:	48 89 ee             	mov    %rbp,%rsi
  401d13:	44 89 e7             	mov    %r12d,%edi
  401d16:	e8 15 ee ff ff       	callq  400b30 <write@plt>
  401d1b:	48 85 c0             	test   %rax,%rax
  401d1e:	7f 0f                	jg     401d2f <submitr+0x3c0>
  401d20:	e8 db ed ff ff       	callq  400b00 <__errno_location@plt>
  401d25:	83 38 04             	cmpl   $0x4,(%rax)
  401d28:	75 12                	jne    401d3c <submitr+0x3cd>
  401d2a:	b8 00 00 00 00       	mov    $0x0,%eax
  401d2f:	48 01 c5             	add    %rax,%rbp
  401d32:	48 29 c3             	sub    %rax,%rbx
  401d35:	75 d6                	jne    401d0d <submitr+0x39e>
  401d37:	4d 85 ed             	test   %r13,%r13
  401d3a:	79 5f                	jns    401d9b <submitr+0x42c>
  401d3c:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  401d43:	3a 20 43 
  401d46:	49 89 07             	mov    %rax,(%r15)
  401d49:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  401d50:	20 75 6e 
  401d53:	49 89 47 08          	mov    %rax,0x8(%r15)
  401d57:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  401d5e:	74 6f 20 
  401d61:	49 89 47 10          	mov    %rax,0x10(%r15)
  401d65:	48 b8 77 72 69 74 65 	movabs $0x6f74206574697277,%rax
  401d6c:	20 74 6f 
  401d6f:	49 89 47 18          	mov    %rax,0x18(%r15)
  401d73:	48 b8 20 74 68 65 20 	movabs $0x7265732065687420,%rax
  401d7a:	73 65 72 
  401d7d:	49 89 47 20          	mov    %rax,0x20(%r15)
  401d81:	41 c7 47 28 76 65 72 	movl   $0x726576,0x28(%r15)
  401d88:	00 
  401d89:	44 89 e7             	mov    %r12d,%edi
  401d8c:	e8 cf ed ff ff       	callq  400b60 <close@plt>
  401d91:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  401d96:	e9 9d 02 00 00       	jmpq   402038 <submitr+0x6c9>
  401d9b:	44 89 64 24 40       	mov    %r12d,0x40(%rsp)
  401da0:	c7 44 24 44 00 00 00 	movl   $0x0,0x44(%rsp)
  401da7:	00 
  401da8:	48 8d 44 24 50       	lea    0x50(%rsp),%rax
  401dad:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  401db2:	ba 00 20 00 00       	mov    $0x2000,%edx
  401db7:	48 8d b4 24 50 20 00 	lea    0x2050(%rsp),%rsi
  401dbe:	00 
  401dbf:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  401dc4:	e8 b4 fa ff ff       	callq  40187d <rio_readlineb>
  401dc9:	48 85 c0             	test   %rax,%rax
  401dcc:	7f 74                	jg     401e42 <submitr+0x4d3>
  401dce:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  401dd5:	3a 20 43 
  401dd8:	49 89 07             	mov    %rax,(%r15)
  401ddb:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  401de2:	20 75 6e 
  401de5:	49 89 47 08          	mov    %rax,0x8(%r15)
  401de9:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  401df0:	74 6f 20 
  401df3:	49 89 47 10          	mov    %rax,0x10(%r15)
  401df7:	48 b8 72 65 61 64 20 	movabs $0x7269662064616572,%rax
  401dfe:	66 69 72 
  401e01:	49 89 47 18          	mov    %rax,0x18(%r15)
  401e05:	48 b8 73 74 20 68 65 	movabs $0x6564616568207473,%rax
  401e0c:	61 64 65 
  401e0f:	49 89 47 20          	mov    %rax,0x20(%r15)
  401e13:	48 b8 72 20 66 72 6f 	movabs $0x73206d6f72662072,%rax
  401e1a:	6d 20 73 
  401e1d:	49 89 47 28          	mov    %rax,0x28(%r15)
  401e21:	41 c7 47 30 65 72 76 	movl   $0x65767265,0x30(%r15)
  401e28:	65 
  401e29:	66 41 c7 47 34 72 00 	movw   $0x72,0x34(%r15)
  401e30:	44 89 e7             	mov    %r12d,%edi
  401e33:	e8 28 ed ff ff       	callq  400b60 <close@plt>
  401e38:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  401e3d:	e9 f6 01 00 00       	jmpq   402038 <submitr+0x6c9>
  401e42:	4c 8d 84 24 50 80 00 	lea    0x8050(%rsp),%r8
  401e49:	00 
  401e4a:	48 8d 4c 24 2c       	lea    0x2c(%rsp),%rcx
  401e4f:	48 8d 94 24 50 60 00 	lea    0x6050(%rsp),%rdx
  401e56:	00 
  401e57:	be 37 2a 40 00       	mov    $0x402a37,%esi
  401e5c:	48 8d bc 24 50 20 00 	lea    0x2050(%rsp),%rdi
  401e63:	00 
  401e64:	b8 00 00 00 00       	mov    $0x0,%eax
  401e69:	e8 82 ed ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  401e6e:	44 8b 44 24 2c       	mov    0x2c(%rsp),%r8d
  401e73:	41 81 f8 c8 00 00 00 	cmp    $0xc8,%r8d
  401e7a:	0f 84 be 00 00 00    	je     401f3e <submitr+0x5cf>
  401e80:	4c 8d 8c 24 50 80 00 	lea    0x8050(%rsp),%r9
  401e87:	00 
  401e88:	b9 88 29 40 00       	mov    $0x402988,%ecx
  401e8d:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  401e94:	be 01 00 00 00       	mov    $0x1,%esi
  401e99:	4c 89 ff             	mov    %r15,%rdi
  401e9c:	b8 00 00 00 00       	mov    $0x0,%eax
  401ea1:	e8 da ed ff ff       	callq  400c80 <__sprintf_chk@plt>
  401ea6:	44 89 e7             	mov    %r12d,%edi
  401ea9:	e8 b2 ec ff ff       	callq  400b60 <close@plt>
  401eae:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  401eb3:	e9 80 01 00 00       	jmpq   402038 <submitr+0x6c9>
  401eb8:	ba 00 20 00 00       	mov    $0x2000,%edx
  401ebd:	48 8d b4 24 50 20 00 	lea    0x2050(%rsp),%rsi
  401ec4:	00 
  401ec5:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  401eca:	e8 ae f9 ff ff       	callq  40187d <rio_readlineb>
  401ecf:	48 85 c0             	test   %rax,%rax
  401ed2:	7f 6a                	jg     401f3e <submitr+0x5cf>
  401ed4:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  401edb:	3a 20 43 
  401ede:	49 89 07             	mov    %rax,(%r15)
  401ee1:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  401ee8:	20 75 6e 
  401eeb:	49 89 47 08          	mov    %rax,0x8(%r15)
  401eef:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  401ef6:	74 6f 20 
  401ef9:	49 89 47 10          	mov    %rax,0x10(%r15)
  401efd:	48 b8 72 65 61 64 20 	movabs $0x6165682064616572,%rax
  401f04:	68 65 61 
  401f07:	49 89 47 18          	mov    %rax,0x18(%r15)
  401f0b:	48 b8 64 65 72 73 20 	movabs $0x6f72662073726564,%rax
  401f12:	66 72 6f 
  401f15:	49 89 47 20          	mov    %rax,0x20(%r15)
  401f19:	48 b8 6d 20 73 65 72 	movabs $0x726576726573206d,%rax
  401f20:	76 65 72 
  401f23:	49 89 47 28          	mov    %rax,0x28(%r15)
  401f27:	41 c6 47 30 00       	movb   $0x0,0x30(%r15)
  401f2c:	44 89 e7             	mov    %r12d,%edi
  401f2f:	e8 2c ec ff ff       	callq  400b60 <close@plt>
  401f34:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  401f39:	e9 fa 00 00 00       	jmpq   402038 <submitr+0x6c9>
  401f3e:	80 bc 24 50 20 00 00 	cmpb   $0xd,0x2050(%rsp)
  401f45:	0d 
  401f46:	0f 85 6c ff ff ff    	jne    401eb8 <submitr+0x549>
  401f4c:	80 bc 24 51 20 00 00 	cmpb   $0xa,0x2051(%rsp)
  401f53:	0a 
  401f54:	0f 85 5e ff ff ff    	jne    401eb8 <submitr+0x549>
  401f5a:	80 bc 24 52 20 00 00 	cmpb   $0x0,0x2052(%rsp)
  401f61:	00 
  401f62:	0f 85 50 ff ff ff    	jne    401eb8 <submitr+0x549>
  401f68:	ba 00 20 00 00       	mov    $0x2000,%edx
  401f6d:	48 8d b4 24 50 20 00 	lea    0x2050(%rsp),%rsi
  401f74:	00 
  401f75:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  401f7a:	e8 fe f8 ff ff       	callq  40187d <rio_readlineb>
  401f7f:	48 85 c0             	test   %rax,%rax
  401f82:	7f 70                	jg     401ff4 <submitr+0x685>
  401f84:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  401f8b:	3a 20 43 
  401f8e:	49 89 07             	mov    %rax,(%r15)
  401f91:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  401f98:	20 75 6e 
  401f9b:	49 89 47 08          	mov    %rax,0x8(%r15)
  401f9f:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  401fa6:	74 6f 20 
  401fa9:	49 89 47 10          	mov    %rax,0x10(%r15)
  401fad:	48 b8 72 65 61 64 20 	movabs $0x6174732064616572,%rax
  401fb4:	73 74 61 
  401fb7:	49 89 47 18          	mov    %rax,0x18(%r15)
  401fbb:	48 b8 74 75 73 20 6d 	movabs $0x7373656d20737574,%rax
  401fc2:	65 73 73 
  401fc5:	49 89 47 20          	mov    %rax,0x20(%r15)
  401fc9:	48 b8 61 67 65 20 66 	movabs $0x6d6f726620656761,%rax
  401fd0:	72 6f 6d 
  401fd3:	49 89 47 28          	mov    %rax,0x28(%r15)
  401fd7:	48 b8 20 73 65 72 76 	movabs $0x72657672657320,%rax
  401fde:	65 72 00 
  401fe1:	49 89 47 30          	mov    %rax,0x30(%r15)
  401fe5:	44 89 e7             	mov    %r12d,%edi
  401fe8:	e8 73 eb ff ff       	callq  400b60 <close@plt>
  401fed:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  401ff2:	eb 44                	jmp    402038 <submitr+0x6c9>
  401ff4:	48 8d b4 24 50 20 00 	lea    0x2050(%rsp),%rsi
  401ffb:	00 
  401ffc:	4c 89 ff             	mov    %r15,%rdi
  401fff:	e8 0c eb ff ff       	callq  400b10 <strcpy@plt>
  402004:	44 89 e7             	mov    %r12d,%edi
  402007:	e8 54 eb ff ff       	callq  400b60 <close@plt>
  40200c:	41 0f b6 17          	movzbl (%r15),%edx
  402010:	b8 4f 00 00 00       	mov    $0x4f,%eax
  402015:	29 d0                	sub    %edx,%eax
  402017:	75 15                	jne    40202e <submitr+0x6bf>
  402019:	41 0f b6 57 01       	movzbl 0x1(%r15),%edx
  40201e:	b8 4b 00 00 00       	mov    $0x4b,%eax
  402023:	29 d0                	sub    %edx,%eax
  402025:	75 07                	jne    40202e <submitr+0x6bf>
  402027:	41 0f b6 47 02       	movzbl 0x2(%r15),%eax
  40202c:	f7 d8                	neg    %eax
  40202e:	85 c0                	test   %eax,%eax
  402030:	0f 95 c0             	setne  %al
  402033:	0f b6 c0             	movzbl %al,%eax
  402036:	f7 d8                	neg    %eax
  402038:	48 8b 8c 24 58 a0 00 	mov    0xa058(%rsp),%rcx
  40203f:	00 
  402040:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
  402047:	00 00 
  402049:	0f 84 0a 01 00 00    	je     402159 <submitr+0x7ea>
  40204f:	e9 00 01 00 00       	jmpq   402154 <submitr+0x7e5>
  402054:	48 b8 45 72 72 6f 72 	movabs $0x52203a726f727245,%rax
  40205b:	3a 20 52 
  40205e:	49 89 07             	mov    %rax,(%r15)
  402061:	48 b8 65 73 75 6c 74 	movabs $0x747320746c757365,%rax
  402068:	20 73 74 
  40206b:	49 89 47 08          	mov    %rax,0x8(%r15)
  40206f:	48 b8 72 69 6e 67 20 	movabs $0x6e6f6320676e6972,%rax
  402076:	63 6f 6e 
  402079:	49 89 47 10          	mov    %rax,0x10(%r15)
  40207d:	48 b8 74 61 69 6e 73 	movabs $0x6e6120736e696174,%rax
  402084:	20 61 6e 
  402087:	49 89 47 18          	mov    %rax,0x18(%r15)
  40208b:	48 b8 20 69 6c 6c 65 	movabs $0x6c6167656c6c6920,%rax
  402092:	67 61 6c 
  402095:	49 89 47 20          	mov    %rax,0x20(%r15)
  402099:	48 b8 20 6f 72 20 75 	movabs $0x72706e7520726f20,%rax
  4020a0:	6e 70 72 
  4020a3:	49 89 47 28          	mov    %rax,0x28(%r15)
  4020a7:	48 b8 69 6e 74 61 62 	movabs $0x20656c6261746e69,%rax
  4020ae:	6c 65 20 
  4020b1:	49 89 47 30          	mov    %rax,0x30(%r15)
  4020b5:	48 b8 63 68 61 72 61 	movabs $0x6574636172616863,%rax
  4020bc:	63 74 65 
  4020bf:	49 89 47 38          	mov    %rax,0x38(%r15)
  4020c3:	66 41 c7 47 40 72 2e 	movw   $0x2e72,0x40(%r15)
  4020ca:	41 c6 47 42 00       	movb   $0x0,0x42(%r15)
  4020cf:	44 89 e7             	mov    %r12d,%edi
  4020d2:	e8 89 ea ff ff       	callq  400b60 <close@plt>
  4020d7:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4020dc:	e9 57 ff ff ff       	jmpq   402038 <submitr+0x6c9>
  4020e1:	48 8d 9c 24 50 20 00 	lea    0x2050(%rsp),%rbx
  4020e8:	00 
  4020e9:	48 8d 84 24 50 40 00 	lea    0x4050(%rsp),%rax
  4020f0:	00 
  4020f1:	50                   	push   %rax
  4020f2:	ff 74 24 20          	pushq  0x20(%rsp)
  4020f6:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
  4020fb:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  402100:	b9 b8 29 40 00       	mov    $0x4029b8,%ecx
  402105:	ba 00 20 00 00       	mov    $0x2000,%edx
  40210a:	be 01 00 00 00       	mov    $0x1,%esi
  40210f:	48 89 df             	mov    %rbx,%rdi
  402112:	b8 00 00 00 00       	mov    $0x0,%eax
  402117:	e8 64 eb ff ff       	callq  400c80 <__sprintf_chk@plt>
  40211c:	b8 00 00 00 00       	mov    $0x0,%eax
  402121:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
  402128:	48 89 df             	mov    %rbx,%rdi
  40212b:	f2 ae                	repnz scas %es:(%rdi),%al
  40212d:	48 89 c8             	mov    %rcx,%rax
  402130:	48 f7 d0             	not    %rax
  402133:	4c 8d 68 ff          	lea    -0x1(%rax),%r13
  402137:	48 83 c4 10          	add    $0x10,%rsp
  40213b:	4c 89 eb             	mov    %r13,%rbx
  40213e:	48 8d ac 24 50 20 00 	lea    0x2050(%rsp),%rbp
  402145:	00 
  402146:	4d 85 ed             	test   %r13,%r13
  402149:	0f 85 be fb ff ff    	jne    401d0d <submitr+0x39e>
  40214f:	e9 47 fc ff ff       	jmpq   401d9b <submitr+0x42c>
  402154:	e8 e7 e9 ff ff       	callq  400b40 <__stack_chk_fail@plt>
  402159:	48 81 c4 68 a0 00 00 	add    $0xa068,%rsp
  402160:	5b                   	pop    %rbx
  402161:	5d                   	pop    %rbp
  402162:	41 5c                	pop    %r12
  402164:	41 5d                	pop    %r13
  402166:	41 5e                	pop    %r14
  402168:	41 5f                	pop    %r15
  40216a:	c3                   	retq   

000000000040216b <init_timeout>:
  40216b:	85 ff                	test   %edi,%edi
  40216d:	74 22                	je     402191 <init_timeout+0x26>
  40216f:	53                   	push   %rbx
  402170:	89 fb                	mov    %edi,%ebx
  402172:	be 4f 18 40 00       	mov    $0x40184f,%esi
  402177:	bf 0e 00 00 00       	mov    $0xe,%edi
  40217c:	e8 1f ea ff ff       	callq  400ba0 <signal@plt>
  402181:	85 db                	test   %ebx,%ebx
  402183:	bf 00 00 00 00       	mov    $0x0,%edi
  402188:	0f 49 fb             	cmovns %ebx,%edi
  40218b:	e8 c0 e9 ff ff       	callq  400b50 <alarm@plt>
  402190:	5b                   	pop    %rbx
  402191:	f3 c3                	repz retq 

0000000000402193 <init_driver>:
  402193:	55                   	push   %rbp
  402194:	53                   	push   %rbx
  402195:	48 83 ec 28          	sub    $0x28,%rsp
  402199:	48 89 fd             	mov    %rdi,%rbp
  40219c:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4021a3:	00 00 
  4021a5:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  4021aa:	31 c0                	xor    %eax,%eax
  4021ac:	be 01 00 00 00       	mov    $0x1,%esi
  4021b1:	bf 0d 00 00 00       	mov    $0xd,%edi
  4021b6:	e8 e5 e9 ff ff       	callq  400ba0 <signal@plt>
  4021bb:	be 01 00 00 00       	mov    $0x1,%esi
  4021c0:	bf 1d 00 00 00       	mov    $0x1d,%edi
  4021c5:	e8 d6 e9 ff ff       	callq  400ba0 <signal@plt>
  4021ca:	be 01 00 00 00       	mov    $0x1,%esi
  4021cf:	bf 1d 00 00 00       	mov    $0x1d,%edi
  4021d4:	e8 c7 e9 ff ff       	callq  400ba0 <signal@plt>
  4021d9:	ba 00 00 00 00       	mov    $0x0,%edx
  4021de:	be 01 00 00 00       	mov    $0x1,%esi
  4021e3:	bf 02 00 00 00       	mov    $0x2,%edi
  4021e8:	e8 a3 ea ff ff       	callq  400c90 <socket@plt>
  4021ed:	85 c0                	test   %eax,%eax
  4021ef:	79 4f                	jns    402240 <init_driver+0xad>
  4021f1:	48 b8 45 72 72 6f 72 	movabs $0x43203a726f727245,%rax
  4021f8:	3a 20 43 
  4021fb:	48 89 45 00          	mov    %rax,0x0(%rbp)
  4021ff:	48 b8 6c 69 65 6e 74 	movabs $0x6e7520746e65696c,%rax
  402206:	20 75 6e 
  402209:	48 89 45 08          	mov    %rax,0x8(%rbp)
  40220d:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402214:	74 6f 20 
  402217:	48 89 45 10          	mov    %rax,0x10(%rbp)
  40221b:	48 b8 63 72 65 61 74 	movabs $0x7320657461657263,%rax
  402222:	65 20 73 
  402225:	48 89 45 18          	mov    %rax,0x18(%rbp)
  402229:	c7 45 20 6f 63 6b 65 	movl   $0x656b636f,0x20(%rbp)
  402230:	66 c7 45 24 74 00    	movw   $0x74,0x24(%rbp)
  402236:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40223b:	e9 0c 01 00 00       	jmpq   40234c <init_driver+0x1b9>
  402240:	89 c3                	mov    %eax,%ebx
  402242:	bf 48 2a 40 00       	mov    $0x402a48,%edi
  402247:	e8 64 e9 ff ff       	callq  400bb0 <gethostbyname@plt>
  40224c:	48 85 c0             	test   %rax,%rax
  40224f:	75 68                	jne    4022b9 <init_driver+0x126>
  402251:	48 b8 45 72 72 6f 72 	movabs $0x44203a726f727245,%rax
  402258:	3a 20 44 
  40225b:	48 89 45 00          	mov    %rax,0x0(%rbp)
  40225f:	48 b8 4e 53 20 69 73 	movabs $0x6e7520736920534e,%rax
  402266:	20 75 6e 
  402269:	48 89 45 08          	mov    %rax,0x8(%rbp)
  40226d:	48 b8 61 62 6c 65 20 	movabs $0x206f7420656c6261,%rax
  402274:	74 6f 20 
  402277:	48 89 45 10          	mov    %rax,0x10(%rbp)
  40227b:	48 b8 72 65 73 6f 6c 	movabs $0x2065766c6f736572,%rax
  402282:	76 65 20 
  402285:	48 89 45 18          	mov    %rax,0x18(%rbp)
  402289:	48 b8 73 65 72 76 65 	movabs $0x6120726576726573,%rax
  402290:	72 20 61 
  402293:	48 89 45 20          	mov    %rax,0x20(%rbp)
  402297:	c7 45 28 64 64 72 65 	movl   $0x65726464,0x28(%rbp)
  40229e:	66 c7 45 2c 73 73    	movw   $0x7373,0x2c(%rbp)
  4022a4:	c6 45 2e 00          	movb   $0x0,0x2e(%rbp)
  4022a8:	89 df                	mov    %ebx,%edi
  4022aa:	e8 b1 e8 ff ff       	callq  400b60 <close@plt>
  4022af:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4022b4:	e9 93 00 00 00       	jmpq   40234c <init_driver+0x1b9>
  4022b9:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
  4022c0:	00 
  4022c1:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
  4022c8:	00 00 
  4022ca:	66 c7 04 24 02 00    	movw   $0x2,(%rsp)
  4022d0:	48 63 50 14          	movslq 0x14(%rax),%rdx
  4022d4:	48 8b 40 18          	mov    0x18(%rax),%rax
  4022d8:	48 8d 7c 24 04       	lea    0x4(%rsp),%rdi
  4022dd:	b9 0c 00 00 00       	mov    $0xc,%ecx
  4022e2:	48 8b 30             	mov    (%rax),%rsi
  4022e5:	e8 d6 e8 ff ff       	callq  400bc0 <__memmove_chk@plt>
  4022ea:	66 c7 44 24 02 2b e3 	movw   $0xe32b,0x2(%rsp)
  4022f1:	ba 10 00 00 00       	mov    $0x10,%edx
  4022f6:	48 89 e6             	mov    %rsp,%rsi
  4022f9:	89 df                	mov    %ebx,%edi
  4022fb:	e8 40 e9 ff ff       	callq  400c40 <connect@plt>
  402300:	85 c0                	test   %eax,%eax
  402302:	79 32                	jns    402336 <init_driver+0x1a3>
  402304:	41 b8 48 2a 40 00    	mov    $0x402a48,%r8d
  40230a:	b9 08 2a 40 00       	mov    $0x402a08,%ecx
  40230f:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  402316:	be 01 00 00 00       	mov    $0x1,%esi
  40231b:	48 89 ef             	mov    %rbp,%rdi
  40231e:	b8 00 00 00 00       	mov    $0x0,%eax
  402323:	e8 58 e9 ff ff       	callq  400c80 <__sprintf_chk@plt>
  402328:	89 df                	mov    %ebx,%edi
  40232a:	e8 31 e8 ff ff       	callq  400b60 <close@plt>
  40232f:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  402334:	eb 16                	jmp    40234c <init_driver+0x1b9>
  402336:	89 df                	mov    %ebx,%edi
  402338:	e8 23 e8 ff ff       	callq  400b60 <close@plt>
  40233d:	66 c7 45 00 4f 4b    	movw   $0x4b4f,0x0(%rbp)
  402343:	c6 45 02 00          	movb   $0x0,0x2(%rbp)
  402347:	b8 00 00 00 00       	mov    $0x0,%eax
  40234c:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  402351:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
  402358:	00 00 
  40235a:	74 05                	je     402361 <init_driver+0x1ce>
  40235c:	e8 df e7 ff ff       	callq  400b40 <__stack_chk_fail@plt>
  402361:	48 83 c4 28          	add    $0x28,%rsp
  402365:	5b                   	pop    %rbx
  402366:	5d                   	pop    %rbp
  402367:	c3                   	retq   

0000000000402368 <driver_post>:
  402368:	53                   	push   %rbx
  402369:	48 89 cb             	mov    %rcx,%rbx
  40236c:	85 d2                	test   %edx,%edx
  40236e:	74 27                	je     402397 <driver_post+0x2f>
  402370:	48 89 f2             	mov    %rsi,%rdx
  402373:	be 58 2a 40 00       	mov    $0x402a58,%esi
  402378:	bf 01 00 00 00       	mov    $0x1,%edi
  40237d:	b8 00 00 00 00       	mov    $0x0,%eax
  402382:	e8 79 e8 ff ff       	callq  400c00 <__printf_chk@plt>
  402387:	66 c7 03 4f 4b       	movw   $0x4b4f,(%rbx)
  40238c:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
  402390:	b8 00 00 00 00       	mov    $0x0,%eax
  402395:	eb 41                	jmp    4023d8 <driver_post+0x70>
  402397:	48 85 ff             	test   %rdi,%rdi
  40239a:	74 2e                	je     4023ca <driver_post+0x62>
  40239c:	80 3f 00             	cmpb   $0x0,(%rdi)
  40239f:	74 29                	je     4023ca <driver_post+0x62>
  4023a1:	48 83 ec 08          	sub    $0x8,%rsp
  4023a5:	51                   	push   %rcx
  4023a6:	49 89 f1             	mov    %rsi,%r9
  4023a9:	41 b8 6f 2a 40 00    	mov    $0x402a6f,%r8d
  4023af:	48 89 f9             	mov    %rdi,%rcx
  4023b2:	4c 89 c2             	mov    %r8,%rdx
  4023b5:	be e3 2b 00 00       	mov    $0x2be3,%esi
  4023ba:	bf 48 2a 40 00       	mov    $0x402a48,%edi
  4023bf:	e8 ab f5 ff ff       	callq  40196f <submitr>
  4023c4:	48 83 c4 10          	add    $0x10,%rsp
  4023c8:	eb 0e                	jmp    4023d8 <driver_post+0x70>
  4023ca:	66 c7 03 4f 4b       	movw   $0x4b4f,(%rbx)
  4023cf:	c6 43 02 00          	movb   $0x0,0x2(%rbx)
  4023d3:	b8 00 00 00 00       	mov    $0x0,%eax
  4023d8:	5b                   	pop    %rbx
  4023d9:	c3                   	retq   
  4023da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000004023e0 <__libc_csu_init>:
  4023e0:	41 57                	push   %r15
  4023e2:	41 56                	push   %r14
  4023e4:	41 89 ff             	mov    %edi,%r15d
  4023e7:	41 55                	push   %r13
  4023e9:	41 54                	push   %r12
  4023eb:	4c 8d 25 1e 1a 20 00 	lea    0x201a1e(%rip),%r12        # 603e10 <__frame_dummy_init_array_entry>
  4023f2:	55                   	push   %rbp
  4023f3:	48 8d 2d 1e 1a 20 00 	lea    0x201a1e(%rip),%rbp        # 603e18 <__init_array_end>
  4023fa:	53                   	push   %rbx
  4023fb:	49 89 f6             	mov    %rsi,%r14
  4023fe:	49 89 d5             	mov    %rdx,%r13
  402401:	4c 29 e5             	sub    %r12,%rbp
  402404:	48 83 ec 08          	sub    $0x8,%rsp
  402408:	48 c1 fd 03          	sar    $0x3,%rbp
  40240c:	e8 af e6 ff ff       	callq  400ac0 <_init>
  402411:	48 85 ed             	test   %rbp,%rbp
  402414:	74 20                	je     402436 <__libc_csu_init+0x56>
  402416:	31 db                	xor    %ebx,%ebx
  402418:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40241f:	00 
  402420:	4c 89 ea             	mov    %r13,%rdx
  402423:	4c 89 f6             	mov    %r14,%rsi
  402426:	44 89 ff             	mov    %r15d,%edi
  402429:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  40242d:	48 83 c3 01          	add    $0x1,%rbx
  402431:	48 39 eb             	cmp    %rbp,%rbx
  402434:	75 ea                	jne    402420 <__libc_csu_init+0x40>
  402436:	48 83 c4 08          	add    $0x8,%rsp
  40243a:	5b                   	pop    %rbx
  40243b:	5d                   	pop    %rbp
  40243c:	41 5c                	pop    %r12
  40243e:	41 5d                	pop    %r13
  402440:	41 5e                	pop    %r14
  402442:	41 5f                	pop    %r15
  402444:	c3                   	retq   
  402445:	90                   	nop
  402446:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40244d:	00 00 00 

0000000000402450 <__libc_csu_fini>:
  402450:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000402454 <_fini>:
  402454:	48 83 ec 08          	sub    $0x8,%rsp
  402458:	48 83 c4 08          	add    $0x8,%rsp
  40245c:	c3                   	retq   
