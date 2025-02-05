# Errata for *PETSc for Partial Differential Equations*

> Not everything in the book is correct!  It follows that reader-submitted corrections and comments are very much appreciated.  Please submit them through the [issues](https://github.com/bueler/p4pdes/issues) or [pull requests](https://github.com/bueler/p4pdes/pulls) tabs at the [github](https://github.com/bueler/p4pdes) site, or by email to the author at `elbueler@alaska.edu`.  Corrections to the example programs themselves will appear as commits in the repository, and then in the [releases](https://github.com/bueler/p4pdes/releases).

The list of errata below shows corrections to the text of the published book, including a couple of notable ones labeled **BAD ONE**.

### Chapter 1

* Page 3: In the program `e.c`, a better way to handle an error code returned from `PetscInitialize()` is to get its output `ierr` and then add the check `if (ierr) return ierr;`  The example programs now do this.

### Chapter 2

* Page 15: In the footnote: "solver" --> "solvers".

* Page 29: In the first complete sentence on the page, for clarity add "the action of" before M^{-1}.

### Chapter 3

* Page 47 **BAD ONE**: In the fourth sentence of the text on this page, substitute "global" for "local".

* Page 50: In the first sentence after Code 3.1, add "the" before "locally owned".

### Chapter 4

* Page 89: In the first sentence after Table 4.3, replace "both" with "either".

### Chapter 5

* Page 97: In the sentence defining the local truncation error, remove the first "O(h^1)".

* Page 106: In the first complete sentence on the page, replace "when" with "if and only if".

* Page 110: In the sentence starting "All of these", replace "latter two runs" with "first two runs".

### Chapter 6

* Page 132 **BAD ONE**: In equations (6.6) and (6.7) replace `u_k[i]` with `u_k[j]` and `u_{k+1}[i]` with `u_{k+1}[j]`.

* Page 148: In the third line from the top, replace "systems" with "system".

* Page 163: In Example 6.9, remove "if cy > 1" from "namely the x-direction if cy > 1"; the condition is clearly unnecessary.

### Chapter 7

* Page 176: In the sixth line from the top, replace "This yields a matrix" with "For d > 1 this yields a matrix".

* Page 177: In the fifth line from the bottom, write "compare" instead of "compare with".

* Page 181: Near the end of the first paragraph, replace "would require" with "requires".

### Chapter 8

* Page 213-214: Sentences are out of order.  Move the last two sentences at the bottom of page 213, which start "Consider also ..." and "The `-ksp_view` ...", to be the second and third sentences on page 214, i.e. just before "See Exercise 8.7."

### Chapter 10

* Page 258: In the second complete sentence on the page, replace "and shows the" with "and the".

* Page 259: In the first sentence replace "we need to" with "we propose to".

* Page 264: In the first sentence of the second full paragraph, replace "Our solver" with "Our default solver".

* Page 266: In the last sentence of the first full paragraph, replace "the example in Chapter 14" with "the examples in Chapters 13 and 14".

* Page 267: In sentence starting "Figure 10.15 ...", replace "KSP iterations, flops, and time" with "KSP iterations and flops".

* Page 271: In the last sentence on the page, replace "see Chapter 14 for" with "see Chapters 13 and 14 for".

* Page 272: In item (i) replace "an" with "a".

### Chapter 11

* Page 294: At the end of the first paragraph in the "Advection results" section, replace "but without a flux limiter" with "but with a smaller final time".  In the very next sentence on the same page, remove "is" from "relatively is insensitive".

* Page 296: In the second sentence on this page replace "limiter imposes" with "limiters impose".

* Page 309: In Exercise 11.1 replace "from g" with "from g and a".

### Chapter 12

* Page 315: In the second sentence of the third paragraph replace "on a 2D" with "on 2D".

### Chapter 14

* Page 355: In the second sentence of the second paragraph replace "so assembling" with "assembling".

* Page 357: In the third sentence of the third paragraph replace "for all q \in L^2" with "for all sufficiently smooth q \in L^2".  (This paragraph should be treated as a sketch of why the inf-sup inequality (14.36) justifies the heuristic (14.35).)

* Page 359: In the second to last sentence on the page replace "and is obligatory" with "while MINRES will not work".

* Page 360: In the last sentence of the second bullet point replace "in [10]" with "in the language of [10]".

### Index

* Page 386: The "Liouville-Bratu equation" index entry is missing two page numbers.  Add pages 93 and 196 to the list.
