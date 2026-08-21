# Python Semantics Checklist for Coding Agents

## Purpose

Use this checklist when writing, reviewing, debugging, or refactoring Python code.
Goal: avoid common semantic mistakes in indexing, mutability, assignment, function calls, iteration, comparison, and imports.

## Core Mental Model

Python code should be reasoned from these principles:

```text
name binding > object copying
mutable object > in-place side effects
half-open intervals > stop excluded
truthiness > explicit boolean conversion not always needed
identity != equality
```

## 1. Indexing and Slicing

### Half-Open Slicing

Python slices use:

```python
a[start:stop]
```

Meaning:

```text
include start
exclude stop
```

So:

```python
a[i : i + 10]
```

accesses indices:

```text
i <= k < i+10
```

It does **not** access `i+10`.

### Range Uses Same Rule

```python
range(n)
```

produces:

```text
0, 1, ..., n-1
```

Not `n`.

### Negative Indexing

```python
a[-1]
```

means:

```text
last element
```

Common patterns:

```python
a[-1]  # last item
a[:-1]  # all except last item
a[-2:]  # last two items
```

### Slice Out-of-Bounds Is Safe

```python
a[100:200]
```

returns:

```python
[]
```

But:

```python
a[100]
```

raises:

```python
IndexError
```

### Reverse Slice

```python
a[::-1]
```

returns a reversed copy.

## 2. Assignment and References

### Assignment Does Not Copy

```python
b = a
```

means:

```text
bind name b to same object as a
```

It does **not** create a new object.

Risk:

```python
a = [1, 2]
b = a
b.append(3)

# a is now [1, 2, 3]
```

### Use Explicit Copy When Needed

```python
b = a.copy()
b = a[:]
```

For nested structures, shallow copy may still share inner objects.

Use:

```python
import copy

b = copy.deepcopy(a)
```

when independent nested objects are required.

## 3. Mutability

### Mutable Objects Can Change In Place

Examples:

```text
list
dict
set
```

```python
a = [1, 2]
a.append(3)
```

### Immutable Objects Cannot Change In Place

Examples:

```text
int
float
str
tuple
frozenset
```

```python
s = "abc"
s[0] = "x"  # TypeError
```

Create a new object instead:

```python
s = "x" + s[1:]
```

## 4. In-Place Methods

Methods that mutate objects usually return:

```python
None
```

Example:

```python
a = [3, 1, 2]
a.sort()
```

Correct.

Avoid:

```python
a = a.sort()
```

because `a` becomes:

```python
None
```

Common in-place methods:

```text
list.append()
list.extend()
list.sort()
list.reverse()
dict.update()
set.add()
```

## 5. Mutable Default Arguments

Avoid mutable default values:

```python
def f(x, bag=[]):
    bag.append(x)
    return bag
```

Problem:

```text
same list reused across calls
```

Use:

```python
def f(x, bag=None):
    if bag is None:
        bag = []
    bag.append(x)
    return bag
```

## 6. Truthiness

Python treats empty values as false.

False-like values:

```text
None
False
0
0.0
""
[]
()
{}
set()
```

Preferred:

```python
if items:
    ...
```

Instead of:

```python
if len(items) > 0:
    ...
```

Use explicit checks when meaning matters:

```python
if x is None:
    ...
```

not:

```python
if not x:
    ...
```

when `0`, `""`, or `[]` are valid values.

## 7. Boolean Operators

`and` and `or` return operands, not always booleans.

```python
value = user_input or default_value
```

Meaning:

```text
if user_input is truthy, use user_input
else use default_value
```

Examples:

```python
"hello" or "world"  # "hello"
"" or "world"  # "world"
"hello" and "world"  # "world"
"" and "world"  # ""
```

## 8. Equality vs Identity

### Equality

```python
a == b
```

means:

```text
same value
```

### Identity

```python
a is b
```

means:

```text
same object
```

Example:

```python
a = [1, 2]
b = [1, 2]

a == b  # True
a is b  # False
```

Use identity for `None`:

```python
if x is None:
    ...
```

Avoid:

```python
if x == None:
    ...
```

## 9. Chained Comparisons

Python supports mathematical comparison chains:

```python
1 < x < 10
```

Meaning:

```python
1 < x and x < 10
```

Preferred for range checks.

## 10. Augmented Assignment

`+=` may mutate in place.

For lists:

```python
a = [1, 2]
b = a

a += [3]

# both a and b see [1, 2, 3]
```

For immutable objects:

```python
x = (1, 2)
x += (3,)
```

creates a new tuple.

Agent rule:

```text
check object mutability before using +=
```

## 11. List Multiplication Trap

Avoid this for nested lists:

```python
matrix = [[0] * 3] * 3
```

Problem:

```text
inner lists share same reference
```

Use:

```python
matrix = [[0] * 3 for _ in range(3)]
```

## 12. Iteration

Prefer direct iteration:

```python
for item in items:
    ...
```

Use `enumerate` when index is needed:

```python
for i, item in enumerate(items):
    ...
```

Avoid unnecessary index loops:

```python
for i in range(len(items)):
    ...
```

unless index arithmetic is required.

## 13. Dictionary Access

Dictionary lookup is by key, not position.

```python
d["name"]
```

Use safe access when key may be missing:

```python
d.get("name")
d.get("name", default_value)
```

Use membership check when needed:

```python
if "name" in d:
    ...
```

## 14. Functions Are Objects

Functions can be:

```text
assigned
passed as arguments
returned from functions
stored in data structures
```

Example:

```python
def square(x):
    return x * x


f = square
f(5)
```

Useful for callbacks, decorators, strategies, and higher-order functions.

## 15. Comprehensions Create New Collections

```python
b = [x * 2 for x in a]
```

creates a new list.

It does not mutate `a`.

Use comprehension for transformation.
Use loops or in-place methods for mutation.

## 16. Import Behavior

A module is usually executed once per Python process, then cached in:

```python
sys.modules
```

Use:

```python
if __name__ == "__main__":
    main()
```

to prevent script logic from running during import.

## Agent Review Checklist

Before finalizing Python code, verify:

```text
[ ] Slice stop index intentionally excluded
[ ] range stop value intentionally excluded
[ ] negative indices are intentional
[ ] slicing vs indexing error behavior considered
[ ] assignment does not accidentally share mutable objects
[ ] copy / deepcopy used when needed
[ ] mutable default arguments avoided
[ ] in-place methods not assigned back accidentally
[ ] truthiness does not hide valid values like 0 or ""
[ ] is used only for identity checks, especially None
[ ] == used for value equality
[ ] += does not create unwanted mutation
[ ] nested list multiplication avoided
[ ] enumerate used when index is needed
[ ] dict keys handled safely
[ ] import side effects controlled with __main__ guard
```

## Final Rule

When uncertain, ask:

```text
Am I creating a new object, or mutating / referencing an existing object?
```

This single question prevents most Python semantic bugs.
