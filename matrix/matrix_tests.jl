using Distributions

A = [
    1 2
    1 2
    2 3
    2 3
]

v = A[sample(axes(A, 1)), :]
w = A[sample(axes(A, 1)), :]

println(v)
println(w)

B = Array{Int32}[A[sample(axes(A, 1)), :] for i=1:2]

println(A)
println(B)