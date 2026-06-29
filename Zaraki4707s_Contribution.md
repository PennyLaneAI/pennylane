I gave the performance impact only 4/10, it's because most of these changes improve the source code much more than they improve runtime.

For example:

range(len(seq)) → enumerate(seq)

Old:

for i in range(len(seq)):
    x = seq[i]

New:

for i, x in enumerate(seq):

The new version is cleaner and avoids explicit indexing, but the runtime difference is usually extremely small in CPython. Both are implemented efficiently in C.

sorted(list(set(x))) → sorted(set(x))

This one does save an unnecessary list allocation:

sorted(list(set(x)))

creates:

a set
a list from that set
then sorted() creates another list

while:

sorted(set(x))

skips step 2.

That's a real improvement, but unless the collection is huge or called frequently, the gain is still small.

Why maintainers may still like the PR

Because software engineering isn't only about speed.

Your changes improve:

readability
maintainability
consistency
linting compliance
removal of redundant operations

Those are valuable.

In fact, if I were a maintainer, I'd view this more as:

"This code is now cleaner and more idiomatic Python"

than

"This code is now significantly faster."

When I'd give 9/10 performance impact

Something like:

for item in data:
    if item not in lst:

becoming

lookup = set(lst)
for item in data:
    if item not in lookup:

because that changes complexity from roughly O(n²) to O(n).

Or removing repeated expensive computations inside hot loops.

Or vectorizing Python loops with NumPy.

Those can produce 10x, 100x, or even 1000x improvements.

Your changes are mostly micro-optimizations and code cleanup, which is why I rated the performance impact modestly while rating the overall contribution much higher. 
