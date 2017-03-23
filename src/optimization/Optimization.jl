module Optimization

using Compat
using ..SensitiveFloats

######################################
# Representation typealias + methods #
######################################

# A collection of celestial objects represented via an implicit parameterization.
# In other words, each element is a vector of coefficients describing a unique
# light source in the implied parameterization.
@compat const Representation{T<:Real} = Vector{Vector{T}}

flatten{T}(rep::Representation{T}) = vcat(rep...)::Vector{T}

function flatten!{T}(x::Vector{T}, rep::Representation{T})
    i = 1
    for r in rep
        for j in eachindex(r)
            x[i] = r[j]
            i += 1
        end
    end
    return x
end

function unflatten!{T}(rep::Representation{T}, x::Vector{T})
    i = 1
    for r in rep
        for j in eachindex(r)
            r[j] = x[i]
            i += 1
        end
    end
    return rep
end

############
# includes #
############

include("constraints.jl")
include("minimization.jl")

end # module
