using Base.Cartesian
import Base.Prehashed

# only line that is changed is that I return an extra argument. The unique indices.

unique_custom(A::AbstractArray; dims::Union{Colon,Integer} = :) = _unique_dims_custom(A, dims)

_unique_dims_custom(A::AbstractArray, dims::Colon) = invoke(unique, Tuple{Any}, A)

@generated function _unique_dims_custom(A::AbstractArray{T,N}, dim::Integer) where {T,N}
    quote
        1 <= dim <= $N || return copy(A)
        hashes = zeros(UInt, axes(A, dim))

        # Compute hash for each row
        k = 0
        @nloops $N i A d->(if d == dim; k = i_d; end) begin
            @inbounds hashes[k] = hash(hashes[k], hash((@nref $N A i)))
        end

        # Collect index of first row for each hash
        uniquerow = similar(Array{Int}, axes(A, dim))
        firstrow = Dict{Prehashed,Int}()
        for k = axes(A, dim)
            uniquerow[k] = get!(firstrow, Prehashed(hashes[k]), k)
        end
        uniquerows = collect(values(firstrow))

        # Check for collisions
        collided = falses(axes(A, dim))
        @inbounds begin
            @nloops $N i A d->(if d == dim
                k = i_d
                j_d = uniquerow[k]
            else
                j_d = i_d
            end) begin
                if (@nref $N A j) != (@nref $N A i)
                    collided[k] = true
                end
            end
        end

        if any(collided)
            nowcollided = similar(BitArray, axes(A, dim))
            while any(collided)
                # Collect index of first row for each collided hash
                empty!(firstrow)
                for j = axes(A, dim)
                    collided[j] || continue
                    uniquerow[j] = get!(firstrow, Prehashed(hashes[j]), j)
                end
                for v in values(firstrow)
                    push!(uniquerows, v)
                end

                # Check for collisions
                fill!(nowcollided, false)
                @nloops $N i A d->begin
                    if d == dim
                        k = i_d
                        j_d = uniquerow[k]
                        (!collided[k] || j_d == k) && continue
                    else
                        j_d = i_d
                    end
                end begin
                    if (@nref $N A j) != (@nref $N A i)
                        nowcollided[k] = true
                    end
                end
                (collided, nowcollided) = (nowcollided, collided)
            end
        end
        
        uniquerows, uniquerow
    end
end



# usage

# aa = [1 2 3; 4 5 6; 7 8 9; 1 2 3; 10 11 12; 4 5 6]
# aa_reduced, uniquerows, uniquerow = unique_custom(aa, dims=1)

# idx2repeat = Dict{Int64,Vector{Int64}}()
# for i in 1:length(uniquerow)
# 	idx2repeat[i] = findall(x-> x == i, uniquerow)
# end
