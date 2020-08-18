""" Implementacao do algoritmo KMeans (Naive) em Julia """

using Distributions
using Random

# Primeiramente vamos criar alguns grupos fictícios
# Vamos definir a média e a variância dos dois grupos
const μ_group_a = 20
const μ_group_b = 30
const μ_group_c = -10
const σ = 1.5
# Instanciamos o tipo de distrbuição
norm_a = Normal(μ_group_a, σ)
norm_b = Normal(μ_group_b, σ)
norm_c = Normal(μ_group_c, σ)
# Finalmente geramos os números para as distribuições instanciadas anteriormente
A = rand(norm_a, 100000, 2)
B = rand(norm_b, 100000, 2)
C = rand(norm_c, 100000, 2)
# A função vcat faz a concatenação vertical das matrizes
X = vcat(A, B, C)

# Funções auxiliares
random_index(X) = sample(axes(X, 1))
euclidean_distance(v, w) = √sum((v-w).^2)

# Encontra o centroid mais similar ao vetor v
function find_closest_centroid!(v, centroids, temp_centroids_vectors)
    min_distance = Inf
    selected_centroid = -1
    for (index, centroid) in enumerate(centroids)
        distance = euclidean_distance(v, centroid)
        if distance < min_distance
            min_distance = distance
            selected_centroid = index
        end
    end
    push!(temp_centroids_vectors[selected_centroid], v)
end

# Calcula os novos centroids
function compute_new_centroids(temp_centroids_vectors, n_centroids)
    new_centroids = []
    for i=1:n_centroids
        centroid_vectors = temp_centroids_vectors[i, :][1]
        new_centroid = sum(centroid_vectors)/length(centroid_vectors)
        push!(new_centroids, new_centroid)
    end
    return new_centroids
end

# Calcula a inercia do movimento dos centroids
function compute_inertia(old_centroids, new_centroids, n_centroids)
    new_inertia = 0
    for i=1:n_centroids
        new_inertia += euclidean_distance(old_centroids[i], new_centroids[i])
    end
    return new_inertia
end

function kmeans(n_groups::Int8)
    centroids = [X[random_index(X), :] for i = 1:n_groups]
    temp_centroids_vectors = []
    inertia = Inf

    while inertia ≥ 0.01
        empty!(temp_centroids_vectors)
        for i=1:n_groups
            push!(temp_centroids_vectors, [])
        end

        for i=axes(X, 1)
            find_closest_centroid!(X[i, :], centroids, temp_centroids_vectors)
        end

        new_centroids = compute_new_centroids(temp_centroids_vectors, n_groups)
        inertia = compute_inertia(centroids, new_centroids, n_groups)
        centroids = new_centroids
        println("Current inertia: ", inertia)
    end
    return centroids
end

final_centroids = kmeans(Int8(3))
println("\nFinal centroids:\n")
display(final_centroids)
println()