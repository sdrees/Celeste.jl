module SensitiveFloats

export SensitiveFloat,
       clear!,
       multiply_sfs!,
       add_scaled_sfs!,
       combine_sfs!,
       add_sources_sf!

"""
A function value and its derivative with respect to its arguments.

Attributes:
  v:  The value
  d:  The derivative with respect to each variable in
      P-dimensional VariationalParams for each of S celestial objects
      in a local_P x local_S matrix.
"""
immutable SensitiveFloat{NumType}
    v::Base.RefValue{NumType}

    # local_P x local_S matrix of gradients
    d::Matrix{NumType}

    local_P::Int64
    local_S::Int64

    has_gradient::Bool

    function (::Type{SensitiveFloat{NumType}}){NumType}(local_P, local_S, has_gradient)
        v = Ref(zero(NumType))
        d = zeros(NumType, local_P * has_gradient, local_S * has_gradient)
        new{NumType}(v, d, local_P, local_S, has_gradient)
    end
end

function SensitiveFloat(local_P::Int64, local_S::Int64,
                        has_gradient::Bool = true)
    return SensitiveFloat{Float64}(local_P, local_S, has_gradient)
end

function SensitiveFloat{NumType <: Number}(prototype_sf::SensitiveFloat{NumType})
    SensitiveFloat{NumType}(prototype_sf.local_P,
                            prototype_sf.local_S,
                            prototype_sf.has_gradient)
end

#########################################################

function clear!{NumType <: Number}(sf::SensitiveFloat{NumType})
    sf.v[] = zero(NumType)
    sf.has_gradient && fill!(sf.d, zero(NumType))
end

"""
Updates sf_result in place with g(sf1, sf2), where
g_d = (g_1, g_2) is the gradient of g

The result is stored in sf_result.  The order is done in such a way that
it can overwrite sf1 or sf2 and still be accurate.
"""
function combine_sfs!{T1 <: Number, T2 <: Number}(
                        sf1::SensitiveFloat{T1},
                        sf2::SensitiveFloat{T1},
                        sf_result::SensitiveFloat{T1},
                        v::T1,
                        g_d::Vector{T2})
    if sf_result.has_gradient
        for ind in eachindex(sf_result.d)
            sf_result.d[ind] = g_d[1] * sf1.d[ind] + g_d[2] * sf2.d[ind]
        end
    end

    sf_result.v[] = v
end

"""
Updates sf1 in place with sf1 * sf2.
"""
function multiply_sfs!{NumType <: Number}(sf1::SensitiveFloat{NumType},
                                          sf2::SensitiveFloat{NumType})
    v = sf1.v[] * sf2.v[]
    g_d = NumType[sf2.v[], sf1.v[]]
    combine_sfs!(sf1, sf2, sf1, v, g_d)
end


"""
Update sf1 in place with (sf1 + scale * sf2).
"""
function add_scaled_sfs!{NumType <: Number}(
                    sf1::SensitiveFloat{NumType},
                    sf2::SensitiveFloat{NumType},
                    scale::AbstractFloat)
    sf1.v[] += scale * sf2.v[]

    @assert sf1.has_gradient == sf2.has_gradient

    if sf1.has_gradient
        LinAlg.BLAS.axpy!(scale, sf2.d, sf1.d)
    end

    true # Set definite return type
end


"""
Adds sf2_s to sf1, where sf1 is sensitive to multiple sources and sf2_s is only
sensitive to source s.
"""
function add_sources_sf!{NumType <: Number}(
                    sf_all::SensitiveFloat{NumType},
                    sf_s::SensitiveFloat{NumType},
                    s::Int)
    sf_all.v[] += sf_s.v[]

    @assert size(sf_all.d, 1) == size(sf_s.d, 1)

    P = sf_all.local_P
    P_shifted = P * (s - 1)

    if sf_all.has_gradient
        @assert size(sf_s.d) == (P, 1)
        @inbounds for s_ind1 in 1:P
            s_all_ind1 = P_shifted + s_ind1
            sf_all.d[s_all_ind1] = sf_all.d[s_all_ind1] + sf_s.d[s_ind1]
        end
    end
end


function zero_sensitive_float_array(NumType::DataType,
                                    local_P::Int,
                                    local_S::Int,
                                    d::Integer...)
    sf_array = Array{SensitiveFloat{NumType}}(d)
    for ind in 1:length(sf_array)
        # Do we always want these arrays to have gradients and hessians?
        sf_array[ind] = SensitiveFloat{NumType}(local_P, local_S, true, true)
    end
    sf_array
end

end
