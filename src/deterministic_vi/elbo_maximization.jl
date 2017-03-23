using ..Optimization: MinConfig, ParameterConstraint, BoxConstraint,
                      SimplexConstraint, ConstraintBatch

#################################################
# type for holding work buffers + configuration #
#################################################

immutable ElboMaxConfig{T,N}
    bvn_bundle::Model.BvnBundle{T}
    min_cfg::MinConfig{T,N}
end

# this is a good chunk size because it divides evenly into `length(CanonicalParams)`
const ELBO_CHUNK_SIZE = 11

function ElboMaxConfig{T}(ea::ElboArgs,
                          vp::VariationalParams{T},
                          bound::VariationalParams{T} = vp[ea.active_sources];
                          loc_width::Float64 = 1e-4,
                          loc_scale::Float64 = 1.0,
                          max_iters::Int = 50,
                          constraints::ConstraintBatch = elbo_constraints(bound, loc_width, loc_scale),
                          optim_options::Options = elbo_optim_options(max_iters=max_iters),
                          trust_region::NewtonTrustRegion = elbo_trust_region())
    bvn_bundle = Model.BvnBundle{T}(ea.psf_K, ea.S)
    min_cfg = MinConfig(bound, constraints, optim_options, trust_region, Val{ELBO_CHUNK_SIZE}())
    return ElboMaxConfig(bvn_bundle, min_cfg)
end

function elbo_constraints{T}(bound::VariationalParams{T},
                             loc_width::Real = 1.0e-4,
                             loc_scale::Real = 1.0)
    n_sources = length(bound)
    boxes = Vector{Vector{ParameterConstraint{BoxConstraint}}}(n_sources)
    simplexes = Vector{Vector{ParameterConstraint{SimplexConstraint}}}(n_sources)
    for src in 1:n_sources
        i1, i2 = ids.u[1], ids.u[2]
        u1, u2 = bound[src][i1], bound[src][i2]
        boxes[src] = [
            ParameterConstraint(BoxConstraint(u1 - loc_width, u1 + loc_width, loc_scale), i1),
            ParameterConstraint(BoxConstraint(u2 - loc_width, u2 + loc_width, loc_scale), i2),
            ParameterConstraint(BoxConstraint(1e-2, 0.99, 1.0), ids.e_dev),
            ParameterConstraint(BoxConstraint(1e-2, 0.99, 1.0), ids.e_axis),
            ParameterConstraint(BoxConstraint(-10.0, 10.0, 1.0), ids.e_angle),
            ParameterConstraint(BoxConstraint(0.10, 70.0, 1.0), ids.e_scale),
            ParameterConstraint(BoxConstraint(-1.0, 10.0, 1.0), ids.r1),
            ParameterConstraint(BoxConstraint(1e-4, 0.10, 1.0), ids.r2),
            ParameterConstraint(BoxConstraint(-10.0, 10.0, 1.0), ids.c1[:, 1]),
            ParameterConstraint(BoxConstraint(-10.0, 10.0, 1.0), ids.c1[:, 2]),
            ParameterConstraint(BoxConstraint(1e-4, 1.0, 1.0), ids.c2[:, 1]),
            ParameterConstraint(BoxConstraint(1e-4, 1.0, 1.0), ids.c2[:, 2])
        ]
        simplexes[src] = [
            ParameterConstraint(SimplexConstraint(0.005, 1.0, 2), ids.a),
            ParameterConstraint(SimplexConstraint(0.01/D, 1.0, D), ids.k[:, 1]),
            ParameterConstraint(SimplexConstraint(0.01/D, 1.0, D), ids.k[:, 2])
        ]
    end
    return ConstraintBatch(boxes, simplexes)
end

function elbo_optim_options(; xtol_abs = 1e-7, ftol_rel = 1e-6, max_iters = 50)
    return Optim.Options(x_tol = xtol_abs, f_tol = ftol_rel, g_tol = 1e-8,
                         iterations = max_iters, store_trace = false,
                         show_trace = false, extended_trace = false)
end

function elbo_trust_region(; initial_delta = 1.0, delta_hat = 1e9)
    return Optim.NewtonTrustRegion(initial_delta = initial_delta,
                                   delta_hat = delta_hat)
end

##########################################
# preallocated ElboIntermediateVariables #
##########################################

const ELBO_VARS_POOL = Vector{ElboIntermediateVariables{Float64}}()

get_elbo_vars() = ELBO_VARS_POOL[Base.Threads.threadid()]

function __init__()
    init_thread_pool!(ELBO_VARS_POOL, () -> ElboIntermediateVariables(Float64, 1, true, true))
end

# explicitly call this for use with compiled system image
__init__()

######################
# objective function #
######################

immutable MaximizableElbo{T} <: Function
    ea::ElboArgs
    vp::VariationalParams{T}
    bvn_bundle::Model.BvnBundle{T}
end

function (f::MaximizableElbo{T})(result::SensitiveFloat{T}, params::VariationalParams{T})
    enforce_active_references!(f.ea, f.vp, params)
    elbo_result = elbo(f.ea, f.vp, get_elbo_vars(), f.bvn_bundle)
    result.v[] = -(elbo_result.v[])
    copy_negative!(result.d, elbo_result.d)
    copy_negative!(result.h, elbo_result.h)
    return result
end

function enforce_active_references!{T}(ea::ElboArgs, dest::VariationalParams{T}, src::VariationalParams{T})
    for i in 1:length(ea.active_sources)
        dest[i] = src[ea.active_sources[i]]
    end
    return nothing
end

# benchmarks indicate this is about ~40% faster than
# `map!(-, dest, src)` where both `dest` and `src`
# were instantiated via `rand(50, 50)`
function copy_negative!(dest, src)
    for i in eachindex(dest)
        dest[i] = -(src[i])
    end
    return dest
end

############################
# maximization entry point #
############################

function maximize!(ea::ElboArgs,
                   vp::VariationalParams{Float64},
                   cfg::ElboMaxConfig = ElboMaxConfig(ea, vp))
    enforce_active_references!(ea, cfg.min_cfg.bound, vp)
    return minimize!(MaximizableElbo(ea, vp, cfg.bvn_bundle), cfg.min_cfg)
end
