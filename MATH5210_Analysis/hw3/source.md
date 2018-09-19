---
title: |
  | Assignment 3
  | Limit Problems (using limit properties)
author: Philip Nelson
date: |
  Wednesday $19^{th}$ of September, 2018
fontsize: 12pt
geometry: margin=1in
header-includes:
- \usepackage{xfrac}
- \newcommand{\qedsym}{\hfill \rule{2mm}{2mm}}

---

# 1.

List the properties of limits discussed in class. Give the citation for these properties in the textbook

**Properties:**

#1. Uniqueness (p 26)

#2. Bounded 2.18 (p 35)

#3. Bounded Away 2.21 (p 36)

#4. Preservation of Inequality 2.9 (p 28)

#5. Bounded and Zero 2.12 (p 29)

#6. Squeeze 2.22 (p 37)

#7. Sum Property 2.10 (p 28)

#8. Polynomial Property 2.17 (p 31)

#9. Product Property 2.13 (p 30)

#10. Quotient Property 2.15 (p 31)

#11. Square Roots 2.17 (p 32)

#12. Convergent Subsequence 2.33 (p 45)

#13. Monotone Convergence 2.25 (p 38)

\pagebreak
# 2.1

**Claim:** $\lim\limits_{n\to\infty} \frac{5n-6}{6n+7} = \frac{5}{6}$

**Proof:** $\lim\limits_{n\to\infty} \frac{5n-6}{6n+7} = \lim\limits_{n\to\infty} \frac{5-\frac{6}{n}}{6+\frac{7}{n}}$

Let $a_n = 5 - \frac{6}{n}$\
Then the $\lim\limits_{n\to\infty} a_n = \lim\limits_{n\to\infty} 5 - \frac{6}{n}$\
Using the sum property of limits,
$\lim\limits_{n\to\infty} 5 - \frac{6}{n} = \lim\limits_{n\to\infty} 5 - \lim\limits_{n\to\infty} \frac{6}{n}$\
From common knowledge and previous $\epsilon - N$ proofs, we know
$\lim\limits_{n\to\infty} 5 = 5$ and $\lim\limits_{n\to\infty} \frac{6}{n} = 0$\
Thus $\lim\limits_{n\to\infty} 5 - \frac{6}{n} = 5 + 0 = 5$

Let $b_n = 6 + \frac{7}{n}$\
Then the $\lim\limits_{n\to\infty} b_n = \lim\limits_{n\to\infty} 6 + \frac{7}{n}$\
Using the sum property of limits,
$\lim\limits_{n\to\infty} 6 - \frac{7}{n} = \lim\limits_{n\to\infty} 6 - \lim\limits_{n\to\infty} \frac{7}{n}$\
From common knowledge and previous $\epsilon - N$ proofs, we know
$\lim\limits_{n\to\infty} 6 = 6$ and $\lim\limits_{n\to\infty} \frac{7}{n} = 0$\
Thus $\lim\limits_{n\to\infty} 6 - \frac{7}{n} = 6 + 0 = 6$

Finally, using the quotient property of limits and  previous two statements
$\lim\limits_{n\to\infty} \frac{5n-6}{6n+7} = \lim\limits_{n\to\infty}\frac{a_n}{b_n} = \frac{5}{6}$ \qedsym{}

# 2.2

**Claim:** $\lim\limits_{n\to\infty} \frac{n^2 + n + 1}{n^2 - 4} = 1$

**Proof:** $\lim\limits_{n\to\infty} \frac{n^2 + n + 1}{n^2 - 4} = \lim\limits_{n\to\infty} \frac{1 + \frac{1}{n} + \frac{1}{n^2}}{1 - \frac{4}{n^2}}$

Let $a_n = 1+\frac{1}{n} + \frac{1}{n^2}$ \
Then $\lim\limits_{n\to\infty} a_n = \lim\limits_{n\to\infty} 1+\frac{1}{n} + \frac{1}{n^2}$ \
Using the sum property of limits,
$\lim\limits_{n\to\infty} 1+\frac{1}{n} + \frac{1}{n^2} = \lim\limits_{n\to\infty} 1 +  \lim\limits_{n\to\infty} \frac{1}{n} +  \lim\limits_{n\to\infty} \frac{1}{n^2}$\
From common knowledge and previous $\epsilon - N$ proofs, we know\
$\lim\limits_{n\to\infty} 1 = 1$, $\lim\limits_{n\to\infty} \frac{1}{n} = 0$, and $\lim\limits_{n\to\infty} \frac{1}{n^2} = 0$\
Thus $\lim\limits_{n\to\infty} 1+\frac{1}{n} + \frac{1}{n^2} = 1 + 0 + 0 = 1$

Let $b_n = 1-\frac{4}{n^2}$ \
Then $\lim\limits_{n\to\infty} b_n = \lim\limits_{n\to\infty} = 1-\frac{4}{n^2}$ \
Using the sum property of limits,
$\lim\limits_{n\to\infty} = 1-\frac{4}{n^2} = \lim\limits_{n\to\infty} 1 - \lim\limits_{n\to\infty} \frac{4}{n^2}$\
From common knowledge and previous $\epsilon - N$ proofs, we know\
$\lim\limits_{n\to\infty} 1 = 1$ and $\lim\limits_{n\to\infty} \frac{4}{n^2} = 0$\
Thus $\lim\limits_{n\to\infty} 1-\frac{4}{n^2} = 1 + 0 = 1$

Finally, using the quotient property of limits and previous two statements
$\lim\limits_{n\to\infty} \frac{n^2 + n + 1}{n^2 - 4} = \frac{1}{1} = 1$

\qedsym{}

# 2.3

**Claim:** $\lim\limits_{n\to\infty} \frac{\sqrt{n^3 + 1}}{n + 2} = \infty$

**Proof:** Begin by multiplying by $\sfrac{\frac{1}{n}}{\frac{1}{n}}$
$$
\lim\limits_{n\to\infty} \frac{\sqrt{n^3 + 1}}{n + 2}
= \lim\limits_{n\to\infty} \frac
{\sqrt{n + \frac{1}{n}}}
{1 + \frac{2}{n}}
$$

By the quotient property of limits
$$
=\frac
{\lim\limits_{n\to\infty}\sqrt{n + \frac{1}{n}}}
{\lim\limits_{n\to\infty}1 + \frac{2}{n}}
$$

Then by the polynomial property of limits
$$
=\frac
{\sqrt{\lim\limits_{n\to\infty}n + \frac{1}{n}}}
{\lim\limits_{n\to\infty}1 + \frac{2}{n}}
$$

Then by the summation property of limits
$$
=\frac
{\sqrt{\lim\limits_{n\to\infty}n + \lim\limits_{n\to\infty}\frac{1}{n}}}
{\lim\limits_{n\to\infty}1 + \lim\limits_{n\to\infty}\frac{2}{n}}
$$

Then by common knowledge and previous proofs
$$
=\frac
{\sqrt{\lim\limits_{n\to\infty}n + 0}}
{1 + 0}
=\frac
{\sqrt{\lim\limits_{n\to\infty}n}}
{1}
=\sqrt{\lim\limits_{n\to\infty}n}
$$

Then by the polynomial property
$$
=\lim\limits_{n\to\infty}\sqrt{n}
$$

Then as previously proven
$$
=\lim\limits_{n\to\infty}\sqrt{n} = \infty
$$

Thus we can see that $\lim\limits_{n\to\infty} \frac{\sqrt{n^3 + 1}}{n + 2} = \infty$

\qedsym{}

# 2.4

**Claim:** $\lim\limits_{n\to\infty} \frac{\sin(n)}{n} = 0$

**Proof:**
We will show by the bounded and zero property of limits that
$\lim\limits_{n\to\infty} \frac{\sin(n)}{n} = 0$

Let $a_n = \sin(n)$     \
As we have previously discussed $|\sin(n)| \leq 1$\
Thus $\sin(n)$ is bounded.

Then let $b_n = \frac{1}{n}$. \
We know from previous $\epsilon - N$ proofs that
$\lim\limits_{n\to\infty} b_n = \lim\limits_{n\to\infty}\frac{1}{n} = 0$

Therefore $\lim\limits_{n\to\infty} \frac{\sin(n)}{n} = 0$ \qedsym

# 2.5

**Claim:** $\lim\limits_{n\to\infty} 1 + 3(-1)^n + 4(-1)^{n+1}$ does not exist

**Proof:** We will show through the convergent sub sequence property that\
$\lim\limits_{n\to\infty} 1 + 3(-1)^n + 4(-1)^{n+1}$ does not exist

Let $a_n = 1 + 3(-1)^n + 4(-1)^{n+1}$

Let $b_n = a_{2n} = 1 - 3 + 4$\
Then the sequence $\{b_n\} = \{ 0, 0, 0, \ldots\}$\
So $\lim\limits_{n\to\infty} b_n = 0$

Let $c_n = a_{2n+1} = 1 + 3 - 4$
Then the sequence $\{c_n\} = \{ 1, 1, 1, \ldots\}$\
So $\lim\limits_{n\to\infty} c_n = 1$

The convergent subsequence property states that if $\lim\limits_{n\to\infty} a_n = A$ then the limit of all subsequence of $a_n$ also converge to $A$. Here we have shown two subsequence $b_n$ and $c_n$ which do not converge to the same value, hence $\lim\limits_{n\to\infty} 1 + 3(-1)^n + 4(-1)^{n+1}$ does not exist.

\qedsym{}

\pagebreak
# 2.6

**Claim:** $\lim\limits_{n\to\infty} \frac{(-1)^n}{n} = 0$

**Proof:**
We will show by the bounded and zero property of limits that
$\lim\limits_{n\to\infty} \frac{(-1)^n}{n} = 0$

Let $a_n = (-1)^n$ \
Then $|a_n| = 1$   \
So $a_n$ is bounded above by $1$.

Then let $b_n = \frac{1}{n}$. \
We know from previous $\epsilon - N$ proofs that
$\lim\limits_{n\to\infty} b_n = \lim\limits_{n\to\infty}\frac{1}{n} = 0$

Therefore $\lim\limits_{n\to\infty} \frac{(-1)^n}{n} = 0$ \qedsym

# 2.7

**Claim:** $\lim\limits_{n\to\infty} \frac{1}{\ln(n)} = 0$

**Proof:** By the squeeze property of limits we will show that
$\lim\limits_{n\to\infty} \frac{1}{\ln(n)} = 0$

Let $a_n = \frac{1}{n}$, $b_n = \frac{1}{\ln(n)}$ and $c_n = \frac{1}{\ln( \ln n)}$

As previously shown $2^n > n^2 > n$ for $n>1$. Observe then that $e^n > 2^n > n \Rightarrow e^n > n$.\
Which $\Rightarrow n > \ln n$\
And $\Rightarrow \frac{1}{n} < \frac{1}{\ln n}$

Next, as we just proved $n > \ln n$\
Then $\Rightarrow \ln n > \ln( \ln n)$\
And $\Rightarrow \frac{1}{\ln n} < \frac{1}{\ln( \ln n)}$

Note, from previous proofs we know that $\lim\limits_{n\to\infty} a_n = \lim\limits_{n\to\infty} \frac{1}{n} = 0$
and $\lim\limits_{n\to\infty} c_n = \lim\limits_{n\to\infty} \frac{1}{\ln( \ln n)} = 0$\
So we have two series, $a_n < b_n < c_n$ and since $\lim\limits_{n\to\infty} a_b = 0$
and $\lim\limits_{n\to\infty} c_n = 0$ then $\lim\limits_{n\to\infty} b_n = 0$

\qedsym{}

\pagebreak
# 2.8

**Claim:** $\lim\limits_{n\to\infty} \sin\frac{n\pi}{3}$ does not exist

**Proof:** By the convergent subsequence property we will show that $\lim\limits_{n\to\infty} \sin\frac{n\pi}{3}$ does not exist

Let $a_n = \sin\frac{n\pi}{3}$

Let $b_n = a_{3n} = \sin(n\pi)$\
Then the sequence $\{b_n\} = \{0, 0, 0, \ldots\}$\
So $\lim\limits_{n\to\infty} b_n = 0$

Let $c_n = a_{6n+1} = \sin(2\pi + \frac{\pi}{3})$\
Then the sequence $\{c_n\} = \{\frac{1}{2}, \frac{1}{2}, \frac{1}{2}, \ldots\}$\
So $\lim\limits_{n\to\infty} c_n = \frac{1}{2}$

The convergent subsequence property states that if $\lim\limits_{n\to\infty} a_n = A$ then the limit of all subsequence of $a_n$ also converge to $A$. Here we have shown two subsequence $b_n$ and $c_n$ which do not converge to the same value, hence $\lim\limits_{n\to\infty} \sin\frac{n\pi}{3}$ does not exist.

\qedsym{}

# 2.9

**Claim:** $\lim\limits_{n\to\infty} \frac{2\sin(n) + 3\cos(n)}{\sqrt{n}} = 0$

**Proof:** We will show through the bounded and zero property of limits that\
$\lim\limits_{n\to\infty} \frac{2\sin(n) + 3\cos(n)}{\sqrt{n}} = 0$

First we will show that $|2\sin(n)+3\cos(n)| \leq M$ for all $n$ and thus is bounded.\
By the triangle inequality we know $|2\sin(n)+3\cos(n)| \leq |2\sin(n)| + |3\cos(n)|$\
As previously discussed, $\sin(n)$ and $\cos(n)$ are bounded by $\pm1$\
So $|2\sin(n)| + |3\cos(n)| \leq 2\cdot 1 + 3\cdot 1 = 5$\
Thus $|2\sin(n)+3\cos(n)| \leq 5$ and so it is bounded.

As previously proven, $\lim\limits_{n\to\infty} \frac{1}{\sqrt{n}} = 0$

Finally, by the product of limits property, we have the product of a bounded series and zero which is zero so $\lim\limits_{n\to\infty} \frac{2\sin(n) + 3\cos(n)}{\sqrt{n}} = 0$

\qedsym{}

\pagebreak
# 2.10

**Claim:** $\lim\limits_{n\to\infty} \frac{\sqrt{n + 4}}{\sqrt{n - 12}} = 1$

**Proof:** $\lim\limits_{n\to\infty} \frac{\sqrt{n + 4}}{\sqrt{n - 12}}
= \lim\limits_{n\to\infty} \sqrt{\frac{n + 4}{n - 12}}$

By the polynomial property of limits
$$\lim\limits_{n\to\infty} \sqrt{\frac{n + 4}{n - 12}} = \sqrt{\lim\limits_{n\to\infty} \frac{n + 4}{n - 12}}
$$
Then by the quotient properties of limits\
$$=\sqrt{\frac
{\lim\limits_{n\to\infty}n+4}
{\lim\limits_{n\to\infty}n-12}
}
= \sqrt{\frac
{\lim\limits_{n\to\infty}1+\frac{4}{n}}
{\lim\limits_{n\to\infty}1-\frac{12}{n}}
}$$
Then by the sum property of limits
$$= \sqrt{\frac
{\lim\limits_{n\to\infty}1+\lim\limits_{n\to\infty}\frac{4}{n}}
{\lim\limits_{n\to\infty}1-\lim\limits_{n\to\infty}\frac{12}{n}}
}$$
Then by common knowledge and previous $\epsilon - N$ proofs
$$= \sqrt{\frac
{1+0}
{1-0}
}
=\sqrt{\frac{1}{1}} = \sqrt{1} = 1$$

Thus we have shown that $\lim\limits_{n\to\infty} \frac{\sqrt{n + 4}}{\sqrt{n - 12}} = 1$

\qedsym{}

\pagebreak
# 2.11

**Claim:** $\lim\limits_{n\to\infty} \sqrt{n^2 + n + 1} - \sqrt{n^2 - n} = 1$

**Proof:**
Using the relation $\sqrt{a} - \sqrt{b}= \frac{a-b}{\sqrt{a}+\sqrt{b}}$
$$\lim\limits_{n\to\infty} \sqrt{n^2 + n + 1} - \sqrt{n^2 - n}
= \lim\limits_{n\to\infty}\frac
{n^2+n+1-n^2+n}
{\sqrt{n^2+n+1} + \sqrt{n^2-n}}
=\lim\limits_{n\to\infty}\frac
{2n+1}
{\sqrt{n^2+n+1} + \sqrt{n^2-n}}
$$

Then multiplying by $\sfrac{\frac{1}{n}}{\frac{1}{n}}$
$$=\lim\limits_{n\to\infty}\frac
{2+\frac{1}{n}}
{\sqrt{1+\frac{1}{n}+\frac{1}{n^2}} + \sqrt{1-\frac{1}{n}}}
$$

Then by the quotient property of limits
$$
=\frac
{\lim\limits_{n\to\infty}2+\frac{1}{n}}
{\lim\limits_{n\to\infty}\sqrt{1+\frac{1}{n}+\frac{1}{n^2}} + \sqrt{1-\frac{1}{n}}}
$$

Then by the summation property of limits
$$
=\frac
{\lim\limits_{n\to\infty}2+\lim\limits_{n\to\infty}\frac{1}{n}}
{\lim\limits_{n\to\infty}\sqrt{1+\frac{1}{n}+\frac{1}{n^2}} + \lim\limits_{n\to\infty}\sqrt{1-\frac{1}{n}}}
$$

Then by the polynomial property of limits
$$
=\frac
{\lim\limits_{n\to\infty}2+\lim\limits_{n\to\infty}\frac{1}{n}}
{\sqrt{\lim\limits_{n\to\infty}1+\frac{1}{n}+\frac{1}{n^2}} + \sqrt{\lim\limits_{n\to\infty}1-\frac{1}{n}}}
$$

Then by the summation property of limits
$$
=\frac
{\lim\limits_{n\to\infty}2+\lim\limits_{n\to\infty}\frac{1}{n}}
{\sqrt{\lim\limits_{n\to\infty}1+\lim\limits_{n\to\infty}\frac{1}{n}+\lim\limits_{n\to\infty}\frac{1}{n^2}} + \sqrt{\lim\limits_{n\to\infty}1-\lim\limits_{n\to\infty}\frac{1}{n}}}
$$

Then by common knowledge and previous proofs
$$
=\frac
{2+0}
{\sqrt{1+0+0}+\sqrt{1-0}}
=\frac{2}{\sqrt{1}+\sqrt{1}}
=\frac{2}{1+1}
=\frac{2}{2}
=1
$$

\qedsym{}
