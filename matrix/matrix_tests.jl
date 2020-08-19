using Distributions

A = [NaN, NaN]
v = [1; 2]
w = [4; 1]

A = hcat(A, v, w)[:, 2:end]

display(A)

B = [NaN NaN]
x = [1 2]
y = [4 1]

B = vcat(B, x, y)[2:end, :]
display(B)