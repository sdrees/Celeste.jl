using Optim
using Optim: Options, NewtonTrustRegion

#################################################
# type for holding work buffers + configuration #
#################################################

immutable MinConfig{T,N}
    bound::Representation{T}
    free::Representation{T}
    bound_result::SensitiveFloat{T}
    free_result::SensitiveFloat{T}
    free_initial_input::Vector{T}
    free_previous_input::Vector{T}
    constraints::ConstraintBatch
    derivs::TransformDerivatives{N,T}
    optim_options::Options{Void}
    trust_region::NewtonTrustRegion{T}
end

function MinConfig(bound::Representation{T},
                   constraints::ConstraintBatch,
                   optim_options::Options{Void},
                   trust_region::NewtonTrustRegion{T},
                   chunk_size::Val)
    free = allocate_free(bound, constraints)
    free_initial_input = flatten(free)
    free_previous_input = similar(free_initial_input)
    free_result = SensitiveFloat{Float64}(length(free[1]), length(bound), true, true)
    derivs = TransformDerivatives(bound, free, chunk_size)
    return MinConfig(bound, free,
                     bound_result, free_result,
                     free_initial_input, free_previous_input,
                     constraints, derivs,
                     optim_options, trust_region)
end

##################################
# Callable Types Passed to Optim #
##################################

function evaluate!{F}(f!::F, cfg::MinConfig, x::Vector)
    if x != cfg.free_previous_input
        copy!(cfg.free_previous_input, x)
        unflatten!(cfg.free, x)
        to_bound!(cfg.bound, cfg.free, cfg.constraints)
        f!(cfg.bound_result, cfg.bound)
        propagate_derivatives!(to_bound!, cfg.bound_result,
                               cfg.free_result, cfg.free,
                               cfg.constraints, cfg.derivs)
    end
    return nothing
end

# Objective #
#-----------#

immutable Objective{F,T,N} <: Function
    f::F
    cfg::MinConfig{T,N}
end

function (obj::Objective)(x::Vector)
    evaluate!(obj.f, obj.cfg, x)
    return obj.cfg.free_result.v[]
end

# Gradient #
#----------#

immutable Gradient{F,T,N} <: Function
    f::F
    cfg::MinConfig{T,N}
end

function (grad::Gradient)(x::Vector, gradient_result::Vector)
    evaluate!(grad.f, grad.cfg, x)
    copy!(gradient_result, grad.cfg.free_result.d)
    return gradient_result
end

# Hessian #
#---------#

immutable Hessian{F,T,N} <: Function
    f::F
    cfg::MinConfig{T,N}
end

function (hess::Hessian)(x::Vector, hessian_result::Matrix)
    evaluate!(hess.f, hess.cfg, x)
    copy!(hessian_result, hess.cfg.free_result.h)
    return hessian_result
end

#############
# minimize! #
#############

function minimize!{F,T}(f::F, cfg::MinConfig{T})
    enforce!(cfg.bound, cfg.constraints)
    to_free!(cfg.free, cfg.bound, cfg.constraints)
    flatten!(cfg.free_initial_input, cfg.free)
    fill!(cfg.free_previous_input, NaN)
    R = Optim.MultivariateOptimizationResults{T,1,Optim.NewtonTrustRegion{T}}
    result::R = Optim.optimize(Objective(f, cfg), Gradient(f, cfg), Hessian(f, cfg),
                               cfg.free_initial_input, cfg.trust_region, cfg.optim_options)
    min_value::T = Optim.minimum(result)
    min_solution::Vector{T} = Optim.minimizer(result)
    unflatten!(cfg.free, min_solution)
    to_bound!(cfg.bound, cfg.free, cfg.constraints)
    return result.f_calls, min_value, min_solution, result
end
