using LinearAlgebra

function normalize_columns(A)
    B = copy(A)
    for i in axes(A)[2]
        B[:, i] = B[:, i] / sum(B[:, i])
    end
    return B
end


"""
A primeira implementacao eh a implementacao mais simplista
e sujeita a problemas relacionados a autoreferenciamento
onde o rank de um nó X qualquer é dado pela soma da probabilidade
de se chegar ao nó X através do nó Z, multiplicado pelo rank de Z,
que a principio eh a propria probabilidade de Z.
"""
function iterative_page_rank(A ,iters)
    nodes_qty = size(A)[2]
    rank_array = ones(nodes_qty) / nodes_qty

    for x in 1:iters
        old_rank = copy(rank_array)

        for i in 1:nodes_qty
            new_rank = 0
            for j in 1:nodes_qty
                new_rank += A[i, j] * rank_array[j]
            end
            rank_array[i] = new_rank
        end

        if rank_array == old_rank
            println("Stop on iteration: ", x)
            break
        end
    end

    return rank_array
end

"""
A proxima função inclui um fator de amortecimento, que considera a probabilidade
de alguma usuaria seguir o link de uma pagina ou ir para qualquer outra pagina
aleatoria do grafo.
"""
function iterative_page_rank_with_damping(A ,iters, damping)
    nodes_qty = size(A)[2]
    rank_array = ones(nodes_qty) / nodes_qty

    for x in 1:iters
        old_rank = copy(rank_array)

        for i in 1:nodes_qty
            new_rank = 0
            for j in 1:nodes_qty
                new_rank += damping * A[i, j] * rank_array[j] + (1 - damping)/nodes_qty
            end
            rank_array[i] = new_rank
        end

        if rank_array == old_rank
            println("Stop on iteration: ", x)
            break
        end
    end

    return rank_array
end

"""
Esta versão da função aproveita o uso de operações em matrizes
para simplificar o código e torna-lo mais eficiente.
"""
function linalg_page_rank(A, iters)
    nodes_qty = size(A)[2]
    rank_array = ones(nodes_qty) / nodes_qty
    old_rank = copy(rank_array)
    rank_array = A*rank_array

    x = 0
    while norm(old_rank - rank_array) > 0.01
        old_rank = copy(rank_array)
        rank_array = A*rank_array
        x += 1

        if x == iters
            throw("Failed to converge with $iters iterations")
        end
    end

    return rank_array
end

"""
Assim como a função anterior, esta faz o uso de operações matriciais,
além de adicionar o fator de amortecimento.
"""
function linalg_page_rank_with_damping(A, iters, damping)
    nodes_qty = size(A)[2]
    println(nodes_qty)
    rank_array = ones(nodes_qty) / nodes_qty
    old_rank = copy(rank_array)
    damped_A = (damping * A) + ((1-damping)/nodes_qty) * ones(nodes_qty, nodes_qty)
    rank_array = damped_A * rank_array

    x = 0
    while norm(old_rank - rank_array) > 0.01
        old_rank = copy(rank_array)
        rank_array = damped_A*rank_array
        x += 1

        if x == iters
            throw("Failed to converge with $iters iterations")
        end
    end

    return rank_array
end

A = [
    0. 0. 0. 0. 0.2
    1. 1. 0. 0. 0.2
    0. 0. .5 1. 0.2
    0. 0. 0. 0. 0.2
    0. 0. .5 0. 0.2
]

r = iterative_page_rank(A, 100)
r_d = iterative_page_rank_with_damping(A, 100, .5)
r_ln = linalg_page_rank(A, 20)
r_ln_d = linalg_page_rank_with_damping(A, 20, .9)
