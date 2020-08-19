using LinearAlgebra

"""
Implementacao do processo de Gram-schmidt para geração
de base de vetores ortonormais que geram o espaço vetorial
da matriz A.

Parametros
---
    A: uma matriz de Int64 ou Float64

Resultados
---
    B: a matriz de vetores bases ortonormais

Exemplos
---
julia> A = [1 7; 3 9; 4 8]
3×2 Array{Int64,2}:
 1  7
 3  9
 4  8

 julia> B = gs_basis(A)
 3×2 Array{Float64,2}:
 -0.196116  -0.867315
 -0.588348  -0.269167
 -0.784465   0.418704

Voce pode conferir que o produto entre a matriz B e sua transposta,
equivale a matriz I, assim como esperado para uma matriz de vetores ortonormais.

julia> B'*B
2×2 Array{Float64,2}:
  1.0          -3.37854e-16
 -3.37854e-16   1.0
"""
function gs_basis(A::Union{Array{Float64}, Array{Int64}})
    B::Array{Float64} = copy(A)
    ε::Float64 = 1e-10

    for i=axes(A, 2)
        for j=1:i-1
            B[:, i] = B[:, i] - dot(B[:, i], B[:, j]) * B[:, j]
        end

        norm_b::Float64 = norm(B[:, i])
        B[:, i] = B[:, i] / norm(B[:, i])

        if norm_b ≤ ε
            B[:, i] = zeros(size(B[:, i]))
        end
    end
    return B
end

A = [
    1 2 1 3
    1 0 3 1
    0 2 4 2
    3 1 2 4
]
a_basis = gs_basis(A)

B = [
    2 5 0
    1 1 1
    0 0 1
]
b_basis = gs_basis(B)