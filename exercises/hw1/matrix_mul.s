	.file	"tmpxft_000018c6_00000000-6_matrix_mul.cudafe1.cpp"
	.text
	.local	_ZL22__nv_inited_managed_rt
	.comm	_ZL22__nv_inited_managed_rt,1,1
	.local	_ZL32__nv_fatbinhandle_for_managed_rt
	.comm	_ZL32__nv_fatbinhandle_for_managed_rt,8,8
	.type	_ZL37__nv_save_fatbinhandle_for_managed_rtPPv, @function
_ZL37__nv_save_fatbinhandle_for_managed_rtPPv:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, _ZL32__nv_fatbinhandle_for_managed_rt(%rip)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	_ZL37__nv_save_fatbinhandle_for_managed_rtPPv, .-_ZL37__nv_save_fatbinhandle_for_managed_rtPPv
	.section	.text._ZN4dim3C2Ejjj,"axG",@progbits,_ZN4dim3C5Ejjj,comdat
	.align 2
	.weak	_ZN4dim3C2Ejjj
	.type	_ZN4dim3C2Ejjj, @function
_ZN4dim3C2Ejjj:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movl	%ecx, -20(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %edx
	movl	%edx, (%rax)
	movq	-8(%rbp), %rax
	movl	-16(%rbp), %edx
	movl	%edx, 4(%rax)
	movq	-8(%rbp), %rax
	movl	-20(%rbp), %edx
	movl	%edx, 8(%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	_ZN4dim3C2Ejjj, .-_ZN4dim3C2Ejjj
	.weak	_ZN4dim3C1Ejjj
	.set	_ZN4dim3C1Ejjj,_ZN4dim3C2Ejjj
	.section	.text._ZNSt11char_traitsIcE6lengthEPKc,"axG",@progbits,_ZNSt11char_traitsIcE6lengthEPKc,comdat
	.weak	_ZNSt11char_traitsIcE6lengthEPKc
	.type	_ZNSt11char_traitsIcE6lengthEPKc, @function
_ZNSt11char_traitsIcE6lengthEPKc:
.LFB1984:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	movl	$0, %eax
	testb	%al, %al
	je	.L5
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx11char_traitsIcE6lengthEPKc
	jmp	.L6
.L5:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	nop
.L6:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1984:
	.size	_ZNSt11char_traitsIcE6lengthEPKc, .-_ZNSt11char_traitsIcE6lengthEPKc
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.text._ZN11TimeMonitorC2ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,"axG",@progbits,_ZN11TimeMonitorC5ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,comdat
	.align 2
	.weak	_ZN11TimeMonitorC2ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.type	_ZN11TimeMonitorC2ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, @function
_ZN11TimeMonitorC2ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE:
.LFB3312:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	leaq	16(%rax), %rdx
	movq	-16(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1ERKS4_@PLT
	call	clock@PLT
	movq	-8(%rbp), %rdx
	movq	%rax, (%rdx)
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3312:
	.size	_ZN11TimeMonitorC2ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, .-_ZN11TimeMonitorC2ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.weak	_ZN11TimeMonitorC1ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.set	_ZN11TimeMonitorC1ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,_ZN11TimeMonitorC2ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.section	.rodata
.LC1:
	.string	"TimeMonitor::"
.LC2:
	.string	", took "
.LC3:
	.string	" seconds"
	.section	.text._ZN11TimeMonitorD2Ev,"axG",@progbits,_ZN11TimeMonitorD5Ev,comdat
	.align 2
	.weak	_ZN11TimeMonitorD2Ev
	.type	_ZN11TimeMonitorD2Ev, @function
_ZN11TimeMonitorD2Ev:
.LFB3315:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA3315
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	call	clock@PLT
	movq	-24(%rbp), %rdx
	movq	%rax, 8(%rdx)
	movq	-24(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	subq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%rdx, %xmm0
	movsd	.LC0(%rip), %xmm1
	divsd	%xmm1, %xmm0
	movsd	%xmm0, -8(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rsi
	leaq	_ZSt4cout(%rip), %rax
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	$16, %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__cxx1112basic_stringIS4_S5_T1_EE@PLT
	movq	%rax, %rdx
	leaq	.LC2(%rip), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	movq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	%rax, %xmm0
	movq	%rdx, %rdi
	call	_ZNSolsEd@PLT
	movq	%rax, %rdx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	movq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GOTPCREL(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSolsEPFRSoS_E@PLT
	movq	-24(%rbp), %rax
	addq	$16, %rax
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev@PLT
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3315:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table._ZN11TimeMonitorD2Ev,"aG",@progbits,_ZN11TimeMonitorD5Ev,comdat
.LLSDA3315:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE3315-.LLSDACSB3315
.LLSDACSB3315:
.LLSDACSE3315:
	.section	.text._ZN11TimeMonitorD2Ev,"axG",@progbits,_ZN11TimeMonitorD5Ev,comdat
	.size	_ZN11TimeMonitorD2Ev, .-_ZN11TimeMonitorD2Ev
	.weak	_ZN11TimeMonitorD1Ev
	.set	_ZN11TimeMonitorD1Ev,_ZN11TimeMonitorD2Ev
	.section	.rodata
	.align 4
	.type	_ZL5DSIZE, @object
	.size	_ZL5DSIZE, 4
_ZL5DSIZE:
	.long	4096
	.align 4
	.type	_ZL10block_size, @object
	.size	_ZL10block_size, 4
_ZL10block_size:
	.long	16
	.align 4
	.type	_ZL5A_val, @object
	.size	_ZL5A_val, 4
_ZL5A_val:
	.long	1065353216
	.align 4
	.type	_ZL5B_val, @object
	.size	_ZL5B_val, 4
_ZL5B_val:
	.long	1073741824
	.text
	.globl	_Z9mmul_hostPKfS0_Pfi
	.type	_Z9mmul_hostPKfS0_Pfi, @function
_Z9mmul_hostPKfS0_Pfi:
.LFB3317:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movl	%ecx, -44(%rbp)
	movl	$0, -12(%rbp)
	jmp	.L10
.L15:
	movl	$0, -8(%rbp)
	jmp	.L11
.L14:
	movl	$0, -4(%rbp)
	jmp	.L12
.L13:
	movl	-12(%rbp), %eax
	imull	-44(%rbp), %eax
	movl	%eax, %edx
	movl	-8(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	movl	-12(%rbp), %eax
	imull	-44(%rbp), %eax
	movl	%eax, %edx
	movl	-4(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm2
	movl	-4(%rbp), %eax
	imull	-44(%rbp), %eax
	movl	%eax, %edx
	movl	-8(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	mulss	%xmm2, %xmm0
	movl	-12(%rbp), %eax
	imull	-44(%rbp), %eax
	movl	%eax, %edx
	movl	-8(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -4(%rbp)
.L12:
	movl	-4(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jl	.L13
	addl	$1, -8(%rbp)
.L11:
	movl	-8(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jl	.L14
	addl	$1, -12(%rbp)
.L10:
	movl	-12(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jl	.L15
	nop
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3317:
	.size	_Z9mmul_hostPKfS0_Pfi, .-_Z9mmul_hostPKfS0_Pfi
	.section	.rodata
.LC4:
	.string	"Begin Compute"
.LC8:
	.string	"GPU Compute"
.LC9:
	.string	"matrix_mul.cu"
.LC10:
	.string	"cudaMalloc failure"
	.align 8
.LC11:
	.string	"Fatal error: %s (%s at %s:%d)\n"
.LC12:
	.string	"*** FAILED - ABORTING\n"
.LC13:
	.string	"cudaMemcpy H2D failure"
.LC14:
	.string	"kernel launch failure"
	.align 8
.LC15:
	.string	"kernel execution failure or cudaMemcpy H2D failure"
.LC16:
	.string	"CPU Compute"
	.align 8
.LC19:
	.string	"mismatch at index %d, was: %f, should be: %f\n"
.LC20:
	.string	"Success!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3318:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA3318
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$216, %rsp
	.cfi_offset 3, -24
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	leaq	-124(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIcEC1Ev@PLT
	leaq	-124(%rbp), %rdx
	leaq	-112(%rbp), %rax
	leaq	.LC4(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
.LEHB0:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEPKcRKS3_
.LEHE0:
	leaq	-112(%rbp), %rdx
	leaq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
.LEHB1:
	call	_ZN11TimeMonitorC1ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.LEHE1:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev@PLT
	leaq	-124(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIcED1Ev@PLT
	movl	$67108864, %edi
.LEHB2:
	call	_Znam@PLT
	movq	%rax, -168(%rbp)
	movl	$67108864, %edi
	call	_Znam@PLT
	movq	%rax, -160(%rbp)
	movl	$67108864, %edi
	call	_Znam@PLT
	movq	%rax, -152(%rbp)
	movl	$67108864, %edi
	call	_Znam@PLT
.LEHE2:
	movq	%rax, -144(%rbp)
	movl	$0, -216(%rbp)
	jmp	.L17
.L18:
	movl	-216(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movss	.LC5(%rip), %xmm0
	movss	%xmm0, (%rax)
	movl	-216(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	addq	%rdx, %rax
	movss	.LC6(%rip), %xmm0
	movss	%xmm0, (%rax)
	movl	-216(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-152(%rbp), %rax
	addq	%rdx, %rax
	pxor	%xmm0, %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -216(%rbp)
.L17:
	cmpl	$16777215, -216(%rbp)
	jle	.L18
	leaq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN11TimeMonitorD1Ev
	leaq	-124(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIcEC1Ev@PLT
	leaq	-124(%rbp), %rdx
	leaq	-112(%rbp), %rax
	leaq	.LC8(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
.LEHB3:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEPKcRKS3_
.LEHE3:
	leaq	-112(%rbp), %rdx
	leaq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
.LEHB4:
	call	_ZN11TimeMonitorC1ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.LEHE4:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev@PLT
	leaq	-124(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIcED1Ev@PLT
	leaq	-192(%rbp), %rax
	movl	$67108864, %esi
	movq	%rax, %rdi
.LEHB5:
	call	_Z10cudaMallocIfE9cudaErrorPPT_m
	leaq	-184(%rbp), %rax
	movl	$67108864, %esi
	movq	%rax, %rdi
	call	_Z10cudaMallocIfE9cudaErrorPPT_m
	leaq	-176(%rbp), %rax
	movl	$67108864, %esi
	movq	%rax, %rdi
	call	_Z10cudaMallocIfE9cudaErrorPPT_m
	call	cudaGetLastError@PLT
	movl	%eax, -208(%rbp)
	cmpl	$0, -208(%rbp)
	je	.L19
	movl	-208(%rbp), %eax
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movq	%rax, %rdx
	movq	stderr(%rip), %rax
	movl	$67, %r9d
	leaq	.LC9(%rip), %r8
	movq	%rdx, %rcx
	leaq	.LC10(%rip), %rdx
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$22, %edx
	movl	$1, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L19:
	movq	-192(%rbp), %rax
	movq	-168(%rbp), %rsi
	movl	$1, %ecx
	movl	$67108864, %edx
	movq	%rax, %rdi
	call	cudaMemcpy@PLT
	movq	-184(%rbp), %rax
	movq	-160(%rbp), %rsi
	movl	$1, %ecx
	movl	$67108864, %edx
	movq	%rax, %rdi
	call	cudaMemcpy@PLT
	call	cudaGetLastError@PLT
	movl	%eax, -204(%rbp)
	cmpl	$0, -204(%rbp)
	je	.L20
	movl	-204(%rbp), %eax
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movq	%rax, %rdx
	movq	stderr(%rip), %rax
	movl	$70, %r9d
	leaq	.LC9(%rip), %r8
	movq	%rdx, %rcx
	leaq	.LC13(%rip), %rdx
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$22, %edx
	movl	$1, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L20:
	movl	$16, -136(%rbp)
	movl	$16, -132(%rbp)
	movl	$1, -128(%rbp)
	movl	-132(%rbp), %eax
	addl	$4095, %eax
	movl	-132(%rbp), %ebx
	movl	$0, %edx
	divl	%ebx
	movl	%eax, %edi
	movl	-136(%rbp), %eax
	addl	$4095, %eax
	movl	-136(%rbp), %ebx
	movl	$0, %edx
	divl	%ebx
	movl	%eax, %esi
	leaq	-124(%rbp), %rax
	movl	$1, %ecx
	movl	%edi, %edx
	movq	%rax, %rdi
	call	_ZN4dim3C1Ejjj
	movq	-136(%rbp), %rax
	movl	-128(%rbp), %ecx
	movq	%rcx, %rdx
	movq	-124(%rbp), %rdi
	movl	-116(%rbp), %esi
	movl	$0, %r9d
	movl	$0, %r8d
	movq	%rdx, %rcx
	movq	%rax, %rdx
	call	__cudaPushCallConfiguration@PLT
	testl	%eax, %eax
	jne	.L21
	movq	-176(%rbp), %rdx
	movq	-184(%rbp), %rsi
	movq	-192(%rbp), %rax
	movl	$4096, %ecx
	movq	%rax, %rdi
	call	_Z4mmulPKfS0_Pfi
.L21:
	call	cudaGetLastError@PLT
	movl	%eax, -200(%rbp)
	cmpl	$0, -200(%rbp)
	je	.L22
	movl	-200(%rbp), %eax
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movq	%rax, %rdx
	movq	stderr(%rip), %rax
	movl	$78, %r9d
	leaq	.LC9(%rip), %r8
	movq	%rdx, %rcx
	leaq	.LC14(%rip), %rdx
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$22, %edx
	movl	$1, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L22:
	movq	-176(%rbp), %rsi
	movq	-152(%rbp), %rax
	movl	$2, %ecx
	movl	$67108864, %edx
	movq	%rax, %rdi
	call	cudaMemcpy@PLT
	call	cudaGetLastError@PLT
	movl	%eax, -196(%rbp)
	cmpl	$0, -196(%rbp)
	je	.L23
	movl	-196(%rbp), %eax
	movl	%eax, %edi
	call	cudaGetErrorString@PLT
	movq	%rax, %rdx
	movq	stderr(%rip), %rax
	movl	$83, %r9d
	leaq	.LC9(%rip), %r8
	movq	%rdx, %rcx
	leaq	.LC15(%rip), %rdx
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$22, %edx
	movl	$1, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
.LEHE5:
	movl	$1, %edi
	call	exit@PLT
.L23:
	leaq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN11TimeMonitorD1Ev
	leaq	-124(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIcEC1Ev@PLT
	leaq	-124(%rbp), %rdx
	leaq	-112(%rbp), %rax
	leaq	.LC16(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
.LEHB6:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEPKcRKS3_
.LEHE6:
	leaq	-112(%rbp), %rdx
	leaq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
.LEHB7:
	call	_ZN11TimeMonitorC1ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.LEHE7:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev@PLT
	leaq	-124(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIcED1Ev@PLT
	movq	-144(%rbp), %rdx
	movq	-160(%rbp), %rsi
	movq	-168(%rbp), %rax
	movl	$4096, %ecx
	movq	%rax, %rdi
	call	_Z9mmul_hostPKfS0_Pfi
	leaq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN11TimeMonitorD1Ev
	movl	$0, -212(%rbp)
	jmp	.L24
.L30:
	movl	-212(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-152(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	ucomiss	.LC17(%rip), %xmm0
	jp	.L49
	ucomiss	.LC17(%rip), %xmm0
	je	.L25
.L49:
	movl	-212(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-152(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	pxor	%xmm2, %xmm2
	cvtss2sd	%xmm0, %xmm2
	movq	%xmm2, %rdx
	movsd	.LC18(%rip), %xmm0
	movl	-212(%rbp), %eax
	movapd	%xmm0, %xmm1
	movq	%rdx, %xmm0
	movl	%eax, %esi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$2, %eax
.LEHB8:
	call	printf@PLT
	movl	$-1, %eax
	jmp	.L31
.L25:
	movl	-212(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-144(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movl	-212(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-152(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	ucomiss	%xmm1, %xmm0
	jp	.L50
	ucomiss	%xmm1, %xmm0
	je	.L28
.L50:
	movl	-212(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-152(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movl	-212(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-144(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	pxor	%xmm3, %xmm3
	cvtss2sd	%xmm1, %xmm3
	movq	%xmm3, %rdx
	movl	-212(%rbp), %eax
	movapd	%xmm0, %xmm1
	movq	%rdx, %xmm0
	movl	%eax, %esi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$2, %eax
	call	printf@PLT
	movl	$-1, %eax
	jmp	.L31
.L28:
	addl	$1, -212(%rbp)
.L24:
	cmpl	$16777215, -212(%rbp)
	jle	.L30
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %eax
.L31:
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L40
	jmp	.L51
.L42:
	endbr64
	movq	%rax, %rbx
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev@PLT
	jmp	.L33
.L41:
	endbr64
	movq	%rax, %rbx
.L33:
	leaq	-124(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIcED1Ev@PLT
	movq	%rbx, %rax
	movq	%rax, %rdi
	call	_Unwind_Resume@PLT
.L43:
	endbr64
	movq	%rax, %rbx
	leaq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN11TimeMonitorD1Ev
	movq	%rbx, %rax
	movq	%rax, %rdi
	call	_Unwind_Resume@PLT
.L45:
	endbr64
	movq	%rax, %rbx
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev@PLT
	jmp	.L36
.L44:
	endbr64
	movq	%rax, %rbx
.L36:
	leaq	-124(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIcED1Ev@PLT
	movq	%rbx, %rax
	movq	%rax, %rdi
	call	_Unwind_Resume@PLT
.L46:
	endbr64
	movq	%rax, %rbx
	leaq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN11TimeMonitorD1Ev
	movq	%rbx, %rax
	movq	%rax, %rdi
	call	_Unwind_Resume@PLT
.L48:
	endbr64
	movq	%rax, %rbx
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev@PLT
	jmp	.L39
.L47:
	endbr64
	movq	%rax, %rbx
.L39:
	leaq	-124(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIcED1Ev@PLT
	movq	%rbx, %rax
	movq	%rax, %rdi
	call	_Unwind_Resume@PLT
.LEHE8:
.L51:
	call	__stack_chk_fail@PLT
.L40:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3318:
	.section	.gcc_except_table,"a",@progbits
.LLSDA3318:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE3318-.LLSDACSB3318
.LLSDACSB3318:
	.uleb128 .LEHB0-.LFB3318
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L41-.LFB3318
	.uleb128 0
	.uleb128 .LEHB1-.LFB3318
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L42-.LFB3318
	.uleb128 0
	.uleb128 .LEHB2-.LFB3318
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L43-.LFB3318
	.uleb128 0
	.uleb128 .LEHB3-.LFB3318
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L44-.LFB3318
	.uleb128 0
	.uleb128 .LEHB4-.LFB3318
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L45-.LFB3318
	.uleb128 0
	.uleb128 .LEHB5-.LFB3318
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L46-.LFB3318
	.uleb128 0
	.uleb128 .LEHB6-.LFB3318
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L47-.LFB3318
	.uleb128 0
	.uleb128 .LEHB7-.LFB3318
	.uleb128 .LEHE7-.LEHB7
	.uleb128 .L48-.LFB3318
	.uleb128 0
	.uleb128 .LEHB8-.LFB3318
	.uleb128 .LEHE8-.LEHB8
	.uleb128 0
	.uleb128 0
.LLSDACSE3318:
	.text
	.size	main, .-main
	.local	_ZZL22____nv_dummy_param_refPvE5__ref
	.comm	_ZZL22____nv_dummy_param_refPvE5__ref,8,8
	.type	_ZL22____nv_dummy_param_refPv, @function
_ZL22____nv_dummy_param_refPv:
.LFB3320:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, _ZZL22____nv_dummy_param_refPvE5__ref(%rip)
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3320:
	.size	_ZL22____nv_dummy_param_refPv, .-_ZL22____nv_dummy_param_refPv
	.section	__nv_module_id,"a"
	.align 8
	.type	_ZL15__module_id_str, @object
	.size	_ZL15__module_id_str, 15
_ZL15__module_id_str:
	.string	"__NV_MODULE_ID"
	.local	_ZL20__cudaFatCubinHandle
	.comm	_ZL20__cudaFatCubinHandle,8,8
	.text
	.type	_ZL26__cudaUnregisterBinaryUtilv, @function
_ZL26__cudaUnregisterBinaryUtilv:
.LFB3321:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	_ZL20__cudaFatCubinHandle(%rip), %rax
	movq	%rax, %rdi
	call	_ZL22____nv_dummy_param_refPv
	movq	_ZL20__cudaFatCubinHandle(%rip), %rax
	movq	%rax, %rdi
	call	__cudaUnregisterFatBinary@PLT
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3321:
	.size	_ZL26__cudaUnregisterBinaryUtilv, .-_ZL26__cudaUnregisterBinaryUtilv
	.type	_ZL32__nv_init_managed_rt_with_modulePPv, @function
_ZL32__nv_init_managed_rt_with_modulePPv:
.LFB3322:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	__cudaInitModule@PLT
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3322:
	.size	_ZL32__nv_init_managed_rt_with_modulePPv, .-_ZL32__nv_init_managed_rt_with_modulePPv
#APP
	.section .nv_fatbin, "a"
.align 8
fatbinData:
.quad 0x00100001ba55ed50,0x0000000000001880,0x0000004801010002,0x0000000000001368
.quad 0x0000000000000000,0x0000003400010007,0x0000000000000000,0x0000000000000011
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x33010102464c457f
.quad 0x0000000000000007,0x0000007c00be0002,0x0000000000000000,0x00000000000012c0
.quad 0x0000000000001040,0x0038004000340534,0x0001000a00400003,0x7472747368732e00
.quad 0x747274732e006261,0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261
.quad 0x666e692e766e2e00,0x2e747865742e006f,0x506c756d6d345a5f,0x6966505f3053664b
.quad 0x666e692e766e2e00,0x756d6d345a5f2e6f,0x505f3053664b506c,0x732e766e2e006966
.quad 0x5a5f2e6465726168,0x664b506c756d6d34,0x2e006966505f3053,0x74736e6f632e766e
.quad 0x345a5f2e30746e61,0x53664b506c756d6d,0x722e006966505f30,0x6f632e766e2e6c65
.quad 0x2e30746e6174736e,0x506c756d6d345a5f,0x6966505f3053664b,0x6c61632e766e2e00
.quad 0x2e0068706172676c,0x6f746f72702e766e,0x766e2e0065707974,0x7463612e6c65722e
.quad 0x68732e00006e6f69,0x2e00626174727473,0x2e00626174727473,0x2e006261746d7973
.quad 0x735f6261746d7973,0x766e2e0078646e68,0x742e006f666e692e,0x6d345a5f2e747865
.quad 0x3053664b506c756d,0x766e2e006966505f,0x5a5f2e6f666e692e,0x664b506c756d6d34
.quad 0x2e006966505f3053,0x65726168732e766e,0x756d6d345a5f2e64,0x505f3053664b506c
.quad 0x2e6c65722e006966,0x74736e6f632e766e,0x345a5f2e30746e61,0x53664b506c756d6d
.quad 0x6e2e006966505f30,0x6174736e6f632e76,0x6d345a5f2e30746e,0x3053664b506c756d
.quad 0x766e2e006966505f,0x6172676c6c61632e,0x702e766e2e006870,0x657079746f746f72
.quad 0x6c65722e766e2e00,0x006e6f697463612e,0x506c756d6d345a5f,0x6966505f3053664b
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0009000300000032,0x0000000000000000,0x0000000000000000,0x00080003000000a2
.quad 0x0000000000000000,0x0000000000000000,0x00060003000000c1,0x0000000000000000
.quad 0x0000000000000000,0x00070003000000dd,0x0000000000000000,0x0000000000000000
.quad 0x00091012000000ec,0x0000000000000000,0x0000000000000b40,0x0000000500082f04
.quad 0x0008120400000020,0x0000000000000005,0x0000000500081104,0x0008120400000000
.quad 0x0000000000000005,0x0000007c00043704,0x00002a0100003001,0x0000000200080a04
.quad 0x001c1903001c0140,0x00000000000c1704,0x0011f00000180003,0x00000000000c1704
.quad 0x0021f00000100002,0x00000000000c1704,0x0021f00000080001,0x00000000000c1704
.quad 0x0021f00000000000,0x00081d0400ff1b03,0x0000002800000010,0x0000009800081c04
.quad 0x0000000000000b08,0x00000000ffffffff,0x00000000fffffffe,0x00000000fffffffd
.quad 0x00000000fffffffc,0x0000000000000073,0x3605002511000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x001cc400e22007f6
.quad 0x4c98078000870001,0xf0c8000002670000,0xf0c8000002270005,0x083fc400efa00751
.quad 0xf0c8000002570002,0xf0c8000002170003,0x4f107f8000370006,0x001fc404fca017f1
.quad 0x4e00028000370005,0x4f107f8000270204,0x5b30029800670000,0x001fb000fe2247f5
.quad 0x4e00018000270203,0x4b6d038005670007,0x5b30019800470202,0x001ff400fc8007ed
.quad 0x4b6d200005670207,0x50b0000000070f00,0xe30000000000000f,0x001fb400fea007f1
.quad 0x4c98078005670006,0x5c9807800ff70012,0x366d038000170607,0x081fc440fe2007fd
.quad 0xe24000009e88000f,0x1c0ffffffff70603,0x3848000000270604,0x001fc800fe2207f1
.quad 0x0400000000370605,0x3829000001e70606,0x5c9807800ff70012,0x001ff400fe0007ed
.quad 0x366c038000370307,0x5c9807800ff70003,0xe24000008708000f,0x001fc440fe2007f1
.quad 0x4c12000005670509,0x4e007f800567020b,0x4f107f8005670208,0x001f8400fe2207f1
.quad 0x3829000001e70011,0x5c9807800ff70012,0x4c18810005270010,0x001fc400fe2007f1
.quad 0x5c9807800ff70003,0x5b6903800ff70907,0x5b3005980087020b,0x001f8c00fe2007f1
.quad 0x4c98078005070007,0x4c98078005170008,0x4c10080005371111,0x001ff400fe0207f6
.quad 0x3848000000270b0c,0x3829000001e70b0b,0xe24000006988000f,0x001fb400fe0007ed
.quad 0x3669038000c7090f,0x50900380e0077007,0xe24000004309000f,0x001fc400fe2007ed
.quad 0x50900380e007f007,0x5c10800000c7070e,0x5c98078001070018,0x001fc8001e4007f2
.quad 0x5c98078001170019,0xeed4200000071813,0x5c10080000b7080f,0x101fc000f6c407f0
.quad 0x5c1080000047101a,0xeed4200000070e1c,0x5c1008000067111b,0x001fd880fe0007f1
.quad 0xeed4200000470e15,0x5c10800000471a10,0xeed4200000870e0a,0x101fc000362407f0
.quad 0x5c10080000671b11,0xeed4200000071a14,0x5c10800000471016,0x001ec880fe0007f6
.quad 0xeed4200000c70e1d,0x5c10080000671117,0xeed420000007100d,0x001f880016c00ff0
.quad 0x5c10800000471618,0xeed4200000071610,0x5c10080000671719,0x101fc000fec407f0
.quad 0x5c1080000047181e,0xeed4200000071811,0x5c1008000067191f,0x101fc482fec007f1
.quad 0xeed4200001870e18,0x5c10800000471e1a,0x5c10080000671f1b,0x001fc800fe000ff6
.quad 0x5c10800000471a16,0x5c10080000671b17,0xeed4200000071a1b,0x001fcc00fea007f0
.quad 0x1c0fffffff070909,0xeed4200000071619,0xf0f0000034370000,0x001fc000f62007f0
.quad 0x5980090001c7131c,0xeed4200001070e12,0x3669038000c7090f,0x001fc000fe6000ed
.quad 0xeed4200000071e13,0xf0f0000034270000,0x59800e0001571415,0x003fc000fca007b1
.quad 0xeed4200001470e1c,0x5c10800000471614,0x59800a8000a70d1e,0x001fc800fe0007b1
.quad 0xeed4200001c70e0a,0x5c10080000671715,0xeed4200002070e0d,0x001fcc0015a007f0
.quad 0x5c10800000471416,0xeed420000007141a,0xf0f0000034470000,0x001fc800fe2007f0
.quad 0x59800f0001d7101f,0xeed4200002470e1e,0x5c10080000671517,0x001fcc00fda001b1
.quad 0xeed420000007161d,0x5c10800000471610,0xf0f0000034470000,0x001fb400fe2007e1
.quad 0x59800f800127111f,0x5c10080000671711,0x5c10800000471012,0x003fc000fe2007f3
.quad 0xf0f0000034370000,0xeed4200000071010,0x59800f8001c71315,0x001fc800fe0007b1
.quad 0xeed4200002870e1f,0x5c10080000671113,0xeed4200002c70e1c,0x001f8c00f66007f0
.quad 0x5c10800000471214,0xeed4200000071212,0x59800a8001871b18,0x005fb400fe2007f0
.quad 0x5c10080000671315,0xeed4200003470e13,0x5c10800000471416,0x001fc000fe2007f3
.quad 0xf0f0000034370000,0xeed4200000071414,0x59800c0000a7191b,0x001fc000fe4007b1
.quad 0xeed4200003070e0a,0x5c10080000671517,0x5c10800000471618,0x001fc000fc6007b3
.quad 0xeed4200000071616,0x59800d8000d71a0d,0x5c10080000671719,0x001eb400fe0007f2
.quad 0xeed4200003870e17,0x5c1080000047181a,0xeed4200000071811,0x001fc400fe0007f3
.quad 0xf0f0000034570000,0x5980068001e71d1d,0xeed4200003c70e0d,0x001f9800f62007f2
.quad 0x5c1008000067191b,0xeed4200000071a1e,0x1c10000004070707,0x001fcc00fec007f1
.quad 0x5c1008000087ff08,0x1c00000001070303,0xf0f0000034470000,0x001fd800fe0007fd
.quad 0x59800e8001f7101d,0x59800e8001c7121d,0xf0f0000034270000,0x001fe000fea007e1
.quad 0x59800e8000a71414,0x5c10800000471a10,0x59800a0001371614,0x001fc000fea107f1
.quad 0x59800a0001771114,0x5c10080000671b11,0x59800a0000d71e12,0x001ff400fda007fd
.quad 0xe2400fffbe01000f,0x366903800047090f,0xe24000002209000f,0x001fc8001ec007f0
.quad 0x5c10800000c70716,0xeed4200000071013,0x5c10080000b70817,0x101fc000f6c407f0
.quad 0x5c10800000471014,0xeed420000007161c,0x5c10080000671115,0x001fd880fe0007f1
.quad 0xeed420000047160d,0x5c1080000047141a,0xeed420000087161d,0x101fc000362407f0
.quad 0x5c1008000067151b,0xeed420000007140a,0x5c10800000471a18,0x000ac880fe0007f6
.quad 0xeed4200000c7161f,0x5c10080000671b19,0xeed4200000071a1e,0x001f8800f6c007f0
.quad 0x5c1080000047180e,0xeed4200000071818,0x5c1008000067190f,0x101fc800fec40ff0
.quad 0x5c10800000470e10,0xeed4200000070e0e,0x5c10080000670f11,0x101fc800fec417f0
.quad 0x5c10800000471014,0xeed4200000071010,0x5c10080000671115,0x001f8800fec027f0
.quad 0x5c1080000047141a,0xeed4200000071414,0x5c1008000067151b,0x001fc000fe6007f5
.quad 0xeed4200000071a0f,0xf0f0000034370000,0x5980090001c7131c,0x001eb400fe0007b1
.quad 0xeed4200001071613,0x1c10000002070707,0xeed4200001471612,0x001ed800fe0007f3
.quad 0xf0f0000034370000,0x59800e0000d70a1c,0xeed420000187160a,0x001fd400f62007f0
.quad 0x59800e0001d71e1d,0xeed4200001c7160d,0xf0f0000034470000,0x001fc400fe2007e1
.quad 0x59800e8001f71818,0x50900380e007f007,0x5c1008000087ff08,0x001fcc00ff4007f1
.quad 0x1c00000000870303,0x1c0fffffff870909,0xf0f0000034270000,0x001fb400fe2007e6
.quad 0x59800c0001370e13,0x5980098001271013,0x5c10800000471a10,0x001fd800fea107f1
.quad 0x5980098000a7140a,0x5c10080000671b11,0x5980050000d70f12,0x081fd800ffa007ed
.quad 0x5b6b20000ff70907,0xe24000001388000f,0x5c10800000c70716,0x001fd880fe0007f1
.quad 0x5c10080000b70817,0x5c1080000047100e,0xeed4200000071010,0x101fc000f62407f0
.quad 0x5c1008000067110f,0xeed420000007160d,0x5c10800000470e14,0x001ec480fe0007f6
.quad 0xeed4200000471613,0x5c10080000670f15,0xeed4200000070e0e,0x001fc000fec007f0
.quad 0x5c10800000471418,0xeed420000087161b,0x5c10080000671519,0x001fc400fe0007b1
.quad 0xeed420000007140a,0x1c0fffffffc70909,0xeed4200000c7161d,0x001f8400f6a007f0
.quad 0x1c10000001070707,0xeed420000007181c,0x5b6b03800ff70907,0x001fcc00ff6007f1
.quad 0x5c1008000087ff08,0x1c00000000470303,0xf0f0000034270000,0x001f8400fea007f1
.quad 0x5980090000d7100d,0x5c10800000471810,0x5980068001370e0d,0x001fc020fec007f7
.quad 0x5c10080000671911,0x5980068001b70a0a,0x5980050001d71c12,0x001ff400fda007fd
.quad 0xe2400fffec80000f,0x5b6b03800ff70507,0xe24000001188000f,0x081fd440fe2207f1
.quad 0x4e00000005670308,0x4f107f8005670309,0x4e00018005670207,0x001f8400fea007f1
.quad 0x5b30041800970303,0x4f107f8005670208,0x3829000001e7030f,0x081fc400fea007f1
.quad 0x5b30039800870207,0x4c1881000527030e,0x3829000001e7070d,0x001f9800fec007f1
.quad 0x4c10080005370f0f,0x4c18810005070707,0x4c10080005170d0d,0x001fc000fe4007f1
.quad 0x5c98078000e70008,0x5c98078000f70009,0x5c9807800077000a,0x001ec400fe4007f1
.quad 0xeed4200000070803,0x5c98078000d7000b,0xeed4200000070a0c,0x001fc400fea007e1
.quad 0x1c0ffffffff70505,0x5c10800000470e0e,0x5b6b03800ff70507,0x001fdc00fec007f1
.quad 0x5c10080000670f0f,0x1c10000000470707,0x5c10080000d7ff0d,0x081fc400ffa107f0
.quad 0x5980090000c70312,0xe2400ffff680000f,0x4e00010005670002,0x081fc400fec207f6
.quad 0x4f107f8005670003,0x5b30011800370000,0x3829000001e70002,0x001fd400fc4007f6
.quad 0x4c18810005470004,0x4c10080005570205,0xeedc200000070412,0x001f8000ffe007ff
.quad 0xe30000000007000f,0xe2400fffff87000f,0x50b0000000070f00,0x001f8000fc0007e0
.quad 0x50b0000000070f00,0x50b0000000070f00,0x50b0000000070f00,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000300000001
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000040,0x00000000000000ec
.quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x000000030000000b
.quad 0x0000000000000000,0x0000000000000000,0x000000000000012c,0x00000000000000fd
.quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x0000000200000013
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000230,0x0000000000000090
.quad 0x0000000500000002,0x0000000000000008,0x0000000000000018,0x7000000000000029
.quad 0x0000000000000000,0x0000000000000000,0x00000000000002c0,0x0000000000000030
.quad 0x0000000000000003,0x0000000000000004,0x0000000000000000,0x7000000000000049
.quad 0x0000000000000040,0x0000000000000000,0x00000000000002f0,0x000000000000007c
.quad 0x0000000900000003,0x0000000000000004,0x0000000000000000,0x70000001000000c1
.quad 0x0000000000000000,0x0000000000000000,0x000000000000036c,0x0000000000000020
.quad 0x0000000000000003,0x0000000000000004,0x0000000000000008,0x7000000b000000dd
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000390,0x0000000000000010
.quad 0x0000000000000000,0x0000000000000008,0x0000000000000008,0x000000010000007f
.quad 0x0000000000000042,0x0000000000000000,0x00000000000003a0,0x000000000000015c
.quad 0x0000000900000000,0x0000000000000004,0x0000000000000000,0x0000000100000032
.quad 0x0000000000000006,0x0000000000000000,0x0000000000000500,0x0000000000000b40
.quad 0x2000000500000003,0x0000000000000020,0x0000000000000000,0x0000000500000006
.quad 0x00000000000012c0,0x0000000000000000,0x0000000000000000,0x00000000000000a8
.quad 0x00000000000000a8,0x0000000000000008,0x0000000500000001,0x00000000000003a0
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000ca0,0x0000000000000ca0
.quad 0x0000000000000008,0x0000000500000001,0x00000000000012c0,0x0000000000000000
.quad 0x0000000000000000,0x00000000000000a8,0x00000000000000a8,0x0000000000000008
.quad 0x0000005801010001,0x0000000000000478,0x0000004800000474,0x0000003400080004
.quad 0x0000000000000000,0x0000000000002011,0x0000000000000000,0x0000000000000a82
.quad 0x0000000000000000,0x0000000000000050,0x0000000000000000,0x1ef300032f2f0a3c
.quad 0x6f69737265762e0a,0x742e0a342e38206e,0x6d73207465677261,0x6464612e0a32355f
.quad 0x7a69735f73736572,0xfd00310a34362065,0x20656c626973691c,0x5f207972746e652e
.quad 0x4b506c756d6d345a,0x286966505f305366,0x206d617261702e0a,0x5f11001e3436752e
.quad 0x00262c305f3f001c,0x3217120026311f11,0x05002632332f0026,0x0a7b0a290a3308f3
.quad 0x72702e206765722e,0x3e393c7025206465,0x203233669500123b,0x10001230333c6625
.quad 0xf200127219001262,0x3c64722520343601,0x6c0a0a0a3b3e3433,0x0018752e22007664
.quad 0x007d5b202c38315f,0x1f00303b5d303e04,0x0030311f07003039,0x3219070030371f01
.quad 0x32312f00ba010030,0x3b5d3303f507002f,0x6f742e617476630a,0x656c61626f6c672e
.quad 0x20391f00062c1100,0x3864002032120500,0xb9006a766f6d0a3b,0x6961746325202c33
.quad 0x2c346d0018782e64,0x35440017746e2520,0x646172001625202c,0x140019732e6f6c2e
.quad 0x723900530200352c,0x0067361500383531,0x0067371400187919,0x1b0067381300170a
.quad 0x00352c3224006779,0x7200ce7210005302,0x2365672e70746573,0x1c008f2c31703500
.quad 0x00442c3225001c32,0x024d726f0a3b3254,0x7009f200392c3323,0x20337025400a3b32
.quad 0x5f5f4c2420617262,0x480a3b395f304242,0x34230048746c2300,0x0200b6311201a52c
.quad 0x30202c3932730282,0x1900490100013066,0x0a0a3b3874004934,0x46301400cc646461
.quad 0x6e610a3b312d7300,0x001839322402b764,0x01180100783b3325,0x331f00382c357033
.quad 0x3242014806090078,0x8935190089202c38,0x62752400d2351000,0x7d00007137140089
.quad 0x726c756d0a3b5700,0x1c3112001d321601,0x4301516469773300,0x341e00242c336472
.quad 0x22016f303223001c,0x3602870200f73b34,0x30322e02ae2c3033,0x341f007235140038
.quad 0x63030011011a00e6,0x0a0a3b34005d0300,0x3700843a342800f3,0x81331100282c3132
.quad 0x00006e02032d0303,0x302f001b5b10007f,0x001d331200001d5d,0x616d666403eb3210
.quad 0x2c3423001a6e722e,0x00b7000043020020,0x0404014f0200fd09,0x351300005c351f01
.quad 0x14010079321f005c,0x007b342b2d007936,0x00450200222c3723,0xac01007b0a008d00
.quad 0x01007b0f00820401,0x007b331f005e3813,0x7b381c007b391501,0x0200220101e90100
.quad 0x007b0a008d000045,0x3519008204033101,0x34322f010b03001c,0x007a313223050097
.quad 0x321500001d5d342f,0x980a013231100097,0x460200232c392300,0x05007c0300aa0000
.quad 0x02a20b0006020239,0x0031363129021e06,0x342d2300062c3723,0x702402b76e13039c
.quad 0x3619037102001b36,0x3522028a34170371,0x003a716523003a3a,0x1d003a3914005502
.quad 0x332505030a043437,0x321f03970200bb2c,0x002b2c352300037b,0x1e037c331800da0b
.quad 0x037d313128003935,0x1200723416057b09,0x020000780f008331,0x32260078341d0106
.quad 0x010e36322706822c,0x72702e0a3a3706ff,0x6f6e2220616d6761,0xe8226c6c6f726e75
.quad 0x9c331f0373030001,0x0f001d3432230102,0x00432c3423050202,0x180401110303900d
.quad 0x33260100b6311f01,0xa20300560000cf32,0x07312d2c01b20005,0xcd381901cd381a02
.quad 0x3a382b00f9371701,0x6e03069d35170160,0xad37120407c80f05,0x7f02000180371f00
.quad 0x0091000180351c00,0x732801ba00003a06,0x1400245b11014474,0x39d000a50501215d
.quad 0x0a0a3b7465720a3a, 0x00000000000a0a7d
.text

#NO_APP
	.section	.nvFatBinSegment,"aw"
	.align 8
	.type	_ZL15__fatDeviceText, @object
	.size	_ZL15__fatDeviceText, 24
_ZL15__fatDeviceText:
	.long	1180844977
	.long	1
	.quad	fatbinData
	.quad	0
	.local	_ZZ30__device_stub__Z4mmulPKfS0_PfiPKfS0_PfiE3__f
	.comm	_ZZ30__device_stub__Z4mmulPKfS0_PfiPKfS0_PfiE3__f,8,8
	.text
	.globl	_Z30__device_stub__Z4mmulPKfS0_PfiPKfS0_Pfi
	.type	_Z30__device_stub__Z4mmulPKfS0_PfiPKfS0_Pfi, @function
_Z30__device_stub__Z4mmulPKfS0_PfiPKfS0_Pfi:
.LFB3343:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movq	%rdi, -104(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movl	%ecx, -124(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, -92(%rbp)
	movl	-92(%rbp), %eax
	cltq
	leaq	-104(%rbp), %rdx
	movq	%rdx, -48(%rbp,%rax,8)
	addl	$1, -92(%rbp)
	movl	-92(%rbp), %eax
	cltq
	leaq	-112(%rbp), %rdx
	movq	%rdx, -48(%rbp,%rax,8)
	addl	$1, -92(%rbp)
	movl	-92(%rbp), %eax
	cltq
	leaq	-120(%rbp), %rdx
	movq	%rdx, -48(%rbp,%rax,8)
	addl	$1, -92(%rbp)
	movl	-92(%rbp), %eax
	cltq
	leaq	-124(%rbp), %rdx
	movq	%rdx, -48(%rbp,%rax,8)
	addl	$1, -92(%rbp)
	leaq	_Z4mmulPKfS0_Pfi(%rip), %rax
	movq	%rax, _ZZ30__device_stub__Z4mmulPKfS0_PfiPKfS0_PfiE3__f(%rip)
	movl	$1, -72(%rbp)
	movl	$1, -68(%rbp)
	movl	$1, -64(%rbp)
	movl	$1, -60(%rbp)
	movl	$1, -56(%rbp)
	movl	$1, -52(%rbp)
	leaq	-80(%rbp), %rcx
	leaq	-88(%rbp), %rdx
	leaq	-60(%rbp), %rsi
	leaq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	__cudaPopCallConfiguration@PLT
	testl	%eax, %eax
	setne	%al
	testb	%al, %al
	jne	.L56
	cmpl	$0, -92(%rbp)
	jne	.L59
	movq	-80(%rbp), %rdi
	movq	-88(%rbp), %rsi
	leaq	-48(%rbp), %rdx
	movl	-92(%rbp), %eax
	cltq
	salq	$3, %rax
	leaq	(%rdx,%rax), %r9
	movq	-60(%rbp), %rcx
	movl	-52(%rbp), %r8d
	movq	-72(%rbp), %rdx
	movl	-64(%rbp), %eax
	pushq	%rdi
	pushq	%rsi
	movq	%rdx, %rsi
	movl	%eax, %edx
	leaq	_Z4mmulPKfS0_Pfi(%rip), %rax
	movq	%rax, %rdi
	call	_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st
	addq	$16, %rsp
	jmp	.L56
.L59:
	movq	-80(%rbp), %rdi
	movq	-88(%rbp), %rsi
	leaq	-48(%rbp), %r9
	movq	-60(%rbp), %rcx
	movl	-52(%rbp), %r8d
	movq	-72(%rbp), %rdx
	movl	-64(%rbp), %eax
	pushq	%rdi
	pushq	%rsi
	movq	%rdx, %rsi
	movl	%eax, %edx
	leaq	_Z4mmulPKfS0_Pfi(%rip), %rax
	movq	%rax, %rdi
	call	_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st
	addq	$16, %rsp
.L56:
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L61
	call	__stack_chk_fail@PLT
.L61:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3343:
	.size	_Z30__device_stub__Z4mmulPKfS0_PfiPKfS0_Pfi, .-_Z30__device_stub__Z4mmulPKfS0_PfiPKfS0_Pfi
	.globl	_Z4mmulPKfS0_Pfi
	.type	_Z4mmulPKfS0_Pfi, @function
_Z4mmulPKfS0_Pfi:
.LFB3344:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	%ecx, -28(%rbp)
	movl	-28(%rbp), %ecx
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rsi
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_Z30__device_stub__Z4mmulPKfS0_PfiPKfS0_Pfi
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3344:
	.size	_Z4mmulPKfS0_Pfi, .-_Z4mmulPKfS0_Pfi
	.local	_ZZL31__nv_cudaEntityRegisterCallbackPPvE5__ref
	.comm	_ZZL31__nv_cudaEntityRegisterCallbackPPvE5__ref,8,8
	.section	.rodata
.LC21:
	.string	"_Z4mmulPKfS0_Pfi"
	.text
	.type	_ZL31__nv_cudaEntityRegisterCallbackPPv, @function
_ZL31__nv_cudaEntityRegisterCallbackPPv:
.LFB3345:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, _ZZL31__nv_cudaEntityRegisterCallbackPPvE5__ref(%rip)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZL37__nv_save_fatbinhandle_for_managed_rtPPv
	movq	-8(%rbp), %rax
	pushq	$0
	pushq	$0
	pushq	$0
	pushq	$0
	movl	$0, %r9d
	movl	$-1, %r8d
	leaq	.LC21(%rip), %rdx
	movq	%rdx, %rcx
	leaq	.LC21(%rip), %rdx
	leaq	_Z4mmulPKfS0_Pfi(%rip), %rsi
	movq	%rax, %rdi
	call	__cudaRegisterFunction@PLT
	addq	$32, %rsp
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3345:
	.size	_ZL31__nv_cudaEntityRegisterCallbackPPv, .-_ZL31__nv_cudaEntityRegisterCallbackPPv
	.type	_ZL24__sti____cudaRegisterAllv, @function
_ZL24__sti____cudaRegisterAllv:
.LFB3346:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -15
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	leaq	_ZL15__fatDeviceText(%rip), %rax
	movq	%rax, %rdi
	call	__cudaRegisterFatBinary@PLT
	movq	%rax, _ZL20__cudaFatCubinHandle(%rip)
	leaq	_ZL31__nv_cudaEntityRegisterCallbackPPv(%rip), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rdx
	movq	_ZL20__cudaFatCubinHandle(%rip), %rax
	movq	%rax, %rdi
	call	*%rdx
	movq	_ZL20__cudaFatCubinHandle(%rip), %rax
	movq	%rax, %rdi
	call	__cudaRegisterFatBinaryEnd@PLT
	leaq	_ZL26__cudaUnregisterBinaryUtilv(%rip), %rax
	movq	%rax, %rdi
	call	atexit@PLT
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3346:
	.size	_ZL24__sti____cudaRegisterAllv, .-_ZL24__sti____cudaRegisterAllv
	.section	.init_array,"aw"
	.align 8
	.quad	_ZL24__sti____cudaRegisterAllv
	.section	.text._ZN9__gnu_cxx11char_traitsIcE2eqERKcS3_,"axG",@progbits,_ZN9__gnu_cxx11char_traitsIcE2eqERKcS3_,comdat
	.weak	_ZN9__gnu_cxx11char_traitsIcE2eqERKcS3_
	.type	_ZN9__gnu_cxx11char_traitsIcE2eqERKcS3_, @function
_ZN9__gnu_cxx11char_traitsIcE2eqERKcS3_:
.LFB3399:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movzbl	(%rax), %edx
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	%al, %dl
	sete	%al
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3399:
	.size	_ZN9__gnu_cxx11char_traitsIcE2eqERKcS3_, .-_ZN9__gnu_cxx11char_traitsIcE2eqERKcS3_
	.section	.text._ZN9__gnu_cxx11char_traitsIcE6lengthEPKc,"axG",@progbits,_ZN9__gnu_cxx11char_traitsIcE6lengthEPKc,comdat
	.align 2
	.weak	_ZN9__gnu_cxx11char_traitsIcE6lengthEPKc
	.type	_ZN9__gnu_cxx11char_traitsIcE6lengthEPKc, @function
_ZN9__gnu_cxx11char_traitsIcE6lengthEPKc:
.LFB3398:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -16(%rbp)
	jmp	.L68
.L69:
	addq	$1, -16(%rbp)
.L68:
	movb	$0, -17(%rbp)
	movq	-40(%rbp), %rdx
	movq	-16(%rbp), %rax
	addq	%rax, %rdx
	leaq	-17(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	_ZN9__gnu_cxx11char_traitsIcE2eqERKcS3_
	xorl	$1, %eax
	testb	%al, %al
	jne	.L69
	movq	-16(%rbp), %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L71
	call	__stack_chk_fail@PLT
.L71:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3398:
	.size	_ZN9__gnu_cxx11char_traitsIcE6lengthEPKc, .-_ZN9__gnu_cxx11char_traitsIcE6lengthEPKc
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD2Ev,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD5Ev,comdat
	.align 2
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD2Ev
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD2Ev, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD2Ev:
.LFB3506:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIcED2Ev@PLT
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3506:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD2Ev, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD2Ev
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD1Ev
	.set	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD1Ev,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD2Ev
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC5IS3_EEPKcRKS3_,comdat
	.align 2
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_:
.LFB3667:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA3667
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$72, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	-56(%rbp), %rbx
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
.LEHB9:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv@PLT
	movq	%rax, %rcx
	movq	-72(%rbp), %rax
	movq	%rax, %rdx
	movq	%rcx, %rsi
	movq	%rbx, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC1EPcRKS3_@PLT
.LEHE9:
	cmpq	$0, -64(%rbp)
	je	.L74
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
.LEHB10:
	call	_ZNSt11char_traitsIcE6lengthEPKc
	movq	-64(%rbp), %rdx
	addq	%rdx, %rax
	jmp	.L75
.L74:
	movl	$1, %eax
.L75:
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rdx
	movq	-64(%rbp), %rcx
	movq	-56(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag
.LEHE10:
	jmp	.L79
.L78:
	endbr64
	movq	%rax, %rbx
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD1Ev
	movq	%rbx, %rax
	movq	%rax, %rdi
.LEHB11:
	call	_Unwind_Resume@PLT
.LEHE11:
.L79:
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L77
	call	__stack_chk_fail@PLT
.L77:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3667:
	.section	.gcc_except_table
.LLSDA3667:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE3667-.LLSDACSB3667
.LLSDACSB3667:
	.uleb128 .LEHB9-.LFB3667
	.uleb128 .LEHE9-.LEHB9
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB10-.LFB3667
	.uleb128 .LEHE10-.LEHB10
	.uleb128 .L78-.LFB3667
	.uleb128 0
	.uleb128 .LEHB11-.LFB3667
	.uleb128 .LEHE11-.LEHB11
	.uleb128 0
	.uleb128 0
.LLSDACSE3667:
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC5IS3_EEPKcRKS3_,comdat
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEPKcRKS3_
	.set	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEPKcRKS3_,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.text
	.type	_Z10cudaMallocIfE9cudaErrorPPT_m, @function
_Z10cudaMallocIfE9cudaErrorPPT_m:
.LFB3669:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cudaMalloc@PLT
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3669:
	.size	_Z10cudaMallocIfE9cudaErrorPPT_m, .-_Z10cudaMallocIfE9cudaErrorPPT_m
	.type	_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st, @function
_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st:
.LFB3670:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rcx, %rax
	movl	%r8d, %ecx
	movq	%r9, -48(%rbp)
	movq	%rsi, -24(%rbp)
	movl	%edx, -16(%rbp)
	movq	%rax, -40(%rbp)
	movl	%ecx, -32(%rbp)
	movq	-48(%rbp), %r8
	movq	-40(%rbp), %rcx
	movl	-32(%rbp), %edi
	movq	-24(%rbp), %rsi
	movl	-16(%rbp), %edx
	movq	-8(%rbp), %rax
	pushq	24(%rbp)
	pushq	16(%rbp)
	movq	%r8, %r9
	movl	%edi, %r8d
	movq	%rax, %rdi
	call	cudaLaunchKernel@PLT
	addq	$16, %rsp
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3670:
	.size	_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st, .-_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st
	.section	.text._ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_,"axG",@progbits,_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_,comdat
	.weak	_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_
	.type	_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_, @function
_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_:
.LFB3802:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	leaq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3802:
	.size	_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_, .-_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_
	.section	.rodata
	.align 8
.LC22:
	.string	"basic_string::_M_construct null not valid"
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag,comdat
	.align 2
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag, @function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag:
.LFB3801:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA3801
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_
	testb	%al, %al
	je	.L87
	movq	-48(%rbp), %rax
	cmpq	-56(%rbp), %rax
	je	.L87
	movl	$1, %eax
	jmp	.L88
.L87:
	movl	$0, %eax
.L88:
	testb	%al, %al
	je	.L89
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
.LEHB12:
	call	_ZSt19__throw_logic_errorPKc@PLT
.L89:
	movq	-56(%rbp), %rdx
	movq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	cmpq	$15, %rax
	jbe	.L90
	leaq	-32(%rbp), %rcx
	movq	-40(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm@PLT
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc@PLT
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm@PLT
.LEHE12:
.L90:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
.LEHB13:
	call	_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv@PLT
.LEHE13:
	movq	%rax, %rcx
	movq	-56(%rbp), %rdx
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_S_copy_charsEPcPKcS7_@PLT
	movq	-32(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
.LEHB14:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm@PLT
.LEHE14:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L93
	jmp	.L96
.L94:
	endbr64
	movq	%rax, %rdi
	call	__cxa_begin_catch@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
.LEHB15:
	call	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@PLT
	call	__cxa_rethrow@PLT
.LEHE15:
.L95:
	endbr64
	movq	%rax, %rbx
	call	__cxa_end_catch@PLT
	movq	%rbx, %rax
	movq	%rax, %rdi
.LEHB16:
	call	_Unwind_Resume@PLT
.LEHE16:
.L96:
	call	__stack_chk_fail@PLT
.L93:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3801:
	.section	.gcc_except_table
	.align 4
.LLSDA3801:
	.byte	0xff
	.byte	0x9b
	.uleb128 .LLSDATT3801-.LLSDATTD3801
.LLSDATTD3801:
	.byte	0x1
	.uleb128 .LLSDACSE3801-.LLSDACSB3801
.LLSDACSB3801:
	.uleb128 .LEHB12-.LFB3801
	.uleb128 .LEHE12-.LEHB12
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB13-.LFB3801
	.uleb128 .LEHE13-.LEHB13
	.uleb128 .L94-.LFB3801
	.uleb128 0x1
	.uleb128 .LEHB14-.LFB3801
	.uleb128 .LEHE14-.LEHB14
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB15-.LFB3801
	.uleb128 .LEHE15-.LEHB15
	.uleb128 .L95-.LFB3801
	.uleb128 0
	.uleb128 .LEHB16-.LFB3801
	.uleb128 .LEHE16-.LEHB16
	.uleb128 0
	.uleb128 0
.LLSDACSE3801:
	.byte	0x1
	.byte	0
	.align 4
	.long	0

.LLSDATT3801:
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag,comdat
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag
	.section	.text._ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_,"axG",@progbits,_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_,comdat
	.weak	_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_
	.type	_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_, @function
_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_:
.LFB3867:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	cmpq	$0, -8(%rbp)
	sete	%al
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3867:
	.size	_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_, .-_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_
	.section	.text._ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_,"axG",@progbits,_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_,comdat
	.weak	_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_
	.type	_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_, @function
_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_:
.LFB3868:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3868:
	.size	_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_, .-_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_
	.section	.text._ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag,"axG",@progbits,_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag,comdat
	.weak	_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag
	.type	_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag, @function
_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag:
.LFB3869:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	subq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3869:
	.size	_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag, .-_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag
	.text
	.type	_Z41__static_initialization_and_destruction_0ii, @function
_Z41__static_initialization_and_destruction_0ii:
.LFB4006:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	cmpl	$1, -4(%rbp)
	jne	.L105
	cmpl	$65535, -8(%rbp)
	jne	.L105
	leaq	_ZStL8__ioinit(%rip), %rax
	movq	%rax, %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	leaq	__dso_handle(%rip), %rax
	movq	%rax, %rdx
	leaq	_ZStL8__ioinit(%rip), %rax
	movq	%rax, %rsi
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rax
	movq	%rax, %rdi
	call	__cxa_atexit@PLT
.L105:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4006:
	.size	_Z41__static_initialization_and_destruction_0ii, .-_Z41__static_initialization_and_destruction_0ii
	.type	_GLOBAL__sub_I__Z9mmul_hostPKfS0_Pfi, @function
_GLOBAL__sub_I__Z9mmul_hostPKfS0_Pfi:
.LFB4007:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$65535, %esi
	movl	$1, %edi
	call	_Z41__static_initialization_and_destruction_0ii
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4007:
	.size	_GLOBAL__sub_I__Z9mmul_hostPKfS0_Pfi, .-_GLOBAL__sub_I__Z9mmul_hostPKfS0_Pfi
	.section	.init_array
	.align 8
	.quad	_GLOBAL__sub_I__Z9mmul_hostPKfS0_Pfi
	.section	.rodata
	.align 8
.LC0:
	.long	0
	.long	1093567616
	.align 4
.LC5:
	.long	1065353216
	.align 4
.LC6:
	.long	1073741824
	.align 4
.LC17:
	.long	1174405120
	.align 8
.LC18:
	.long	0
	.long	1086324736
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-2ubuntu1~20.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
