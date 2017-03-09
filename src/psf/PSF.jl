module PSF

using Compat
using ConstraintTransforms: ConstraintTransforms, ParameterConstraint,
                            BoxConstraint, SimplexConstraint

@compat const ComponentParams{T} = Vector{Vector{T}}

const PSF_TRANSFORM_CHUNK_SIZE = Val{length(PsfParams)}()

#################################################
# defaults for optional arguments to `psf_fit!` #
#################################################

immutable PSFBundle{T}
    bound_params::ComponentParams{T}
    free_params::ComponentParams{T}
    bound_result::SensitiveFloat{T}
    free_result::SensitiveFloat{T}
    jacobian_bundle::TransformJacobianBundle{N,T}
    log_pdf::SensitiveFloat{T}
    pdf::SensitiveFloat{T}
    pixel_value::SensitiveFloat{T}
    squared_error::SensitiveFloat{T}
    function (::Type{PSFBundle{T}}){T}(K::Int, constraints::ConstraintBatch)
        n_bound_params = length(PsfParams)
        bound_params = Vector{T}[Vector{T}(n_bound_params) for _ in 1:K]
        free_params = allocate_free_params(bound_params, constraints)
        n_free_params = length(free_params[1])

        result_bound = SensitiveFloat{T}(n_free_params, n_bound_params, true)
        result_free = SensitiveFloat{T}(n_free_params, n_bound_params, true)

        jacobian_bundle = TransformJacobianBundle(bound_params, free_params, PSF_TRANSFORM_CHUNK_SIZE)

        log_pdf = SensitiveFloat{T}(n_bound_params, 1, true)
        pdf = SensitiveFloat{T}(n_bound_params, 1, true)

        pixel_value = SensitiveFloat{T}(n_bound_params, K, true)
        squared_error = SensitiveFloat{T}(n_bound_params, K, true)
        return new{T}(bound_params, free_params,
                      bound_result, free_result,
                      jacobian_bundle,
                      log_pdf, pdf,
                      pixel_value,
                      squared_error)
    end
end

immutable Config{N,T}
    coordinates::Vector{SVector{2,T}}
    psf_bundle::PSFBundle{T}
    dual_psf_bundle::PSFBundle{Dual{1,T}}
    constraints::ConstraintBatch
    optim_options::Options{Void}
    trust_region::CGTrustRegion{T}
end

function Config{T}(raw_psf::Matrix{T}, init_params::ComponentParams{T};
                   optim_options::Options = psf_optim_options(),
                   trust_region::CGTrustRegion = psf_trust_region())
    coordinates = psf_coordinates(size(raw_psf, 1), size(raw_psf, 2))
    constraints = psf_constraints(init_params)
    K = length(init_params)
    psf_bundle = PSFBundle{T}(K, constraints)
    dual_psf_bundle = PSFBundle{Dual{1,T}}(K, constraints)
    return Config(coordinates, psf_bundle, dual_psf_bundle, constraints, optim_options, trust_region)
end

function psf_constraints{T}(init_params::ComponentParams{T})
    n_components = length(init_params)
    boxes = Vector{Vector{ParameterConstraint{BoxConstraint}}}(n_sources)
    simplexes = Vector{Vector{ParameterConstraint{SimplexConstraint}}}(n_sources)
    for i in 1:n_components
        boxes[i] = [
            ParameterConstraint(BoxConstraint(-5.0, 5.0, 1.0), psf_ids.mu),
            ParameterConstraint(BoxConstraint(0.1, 1.0, 1.0), psf_ids.e_axis),
            ParameterConstraint(BoxConstraint(-4π, 4π, 1.0), psf_ids.e_angle),
            ParameterConstraint(BoxConstraint(0.05, 10.0, 1.0), psf_ids.e_scale),
            ParameterConstraint(BoxConstraint(0.05, 2.0, 1.0), psf_ids.weight)
        ]
        simplexes[i] = Vector{ParameterConstraint{SimplexConstraint}}(0)
    end
    return ConstraintBatch(boxes, simplexes)
end

function psf_coordinates(rows::Int, cols::Int)
    psf_row_center = (rows - 1) / 2 + 1
    psf_col_center = (cols - 1) / 2 + 1
    coordinates = Vector{SVector{2,Float64}}(rows * cols)
    k = 1
    for j in 1:rows, i in 1:cols
        coordinates[k] = SVector{2,Float64}(i - psf_row_center, j - psf_col_center)
        k += 1
    end
    return coordinates
end

function psf_optim_options(; x_tol::Float64 = 0.0, f_tol::Float64 = 1e-9,
                           g_tol::Float64 = 1e-9, iterations::Int = 50,
                           verbose::Bool = false)
    return Optim.Options(; x_tol = x_tol, f_tol = f_tol, g_tol = g_tol,
                         iterations = iterations, store_trace = verbose,
                         extended_trace = verbose, show_trace = false)
end

function psf_trust_region(; initial_delta=10.0, delta_hat=1e9, eta=0.1,
                          rho_lower=0.2, rho_upper=0.75)
    return CGTrustRegion(; initial_delta=initial_delta, delta_hat=delta_hat,
                         eta=eta, rho_lower=rho_lower, rho_upper=rho_upper)
end

# ##########################
# # to_vector/from_vector! #
# ##########################
#
# to_vector{T}(sources::Vector{Vector{T}}) = vcat(sources...)::Vector{T}
#
# function from_vector!(sources, x)
#     i = 1
#     for src in sources
#         for j in eachindex(src)
#             src[j] = x[i]
#             i += 1
#         end
#     end
#     return sources
# end
#
# function dual_from_vector!(sources, x, v)
#     i = 1
#     for src in sources
#         for j in 1:length(src)
#             src[j] = Dual(x[i], v[i])
#             i += 1
#         end
#     end
#     return sources
# end

###############
# PSF Fitting #
###############

function evaluate_psf_fit!{T}(cfg::Config, raw_psf::Matrix{T}, calculate_gradient::Bool)
    sync_bvn_buffers!(cfg)
    SensitiveFloats.clear!(cfg.squared_error)
    for coord_index in 1:length(cfg.coordinates)
        evaluate_psf_pixel_fit!(cfg, raw_psf, calculate_gradient, coord_index)
        diff = (cfg.pixel_value.v[] - raw_psf[coord_index])
        cfg.squared_error.v[] += diff*diff
        if calculate_gradient
            for i in 1:length(cfg.squared_error.d)
                @inbounds cfg.squared_error.d[i] += 2 * diff * cfg.pixel_value.d[i]
            end
        end
    end
    return nothing
end

function evaluate_psf_pixel_fit!(cfg::Config, raw_psf::Matrix{T}, calculate_gradient::Bool, coord_index::Int)
    SensitiveFloats.clear!(cfg.pixel_value)
    coord = cfg.coordinates[coord_index]
    # @inbounds for k = 1:K
    #     bvn = bvn_vec[k]
    #     eval_bvn_pdf!(bvn_derivs, bvn, coord)
    #     get_bvn_derivs!(bvn_derivs, bvn, true, true)
    #     transform_bvn_derivs!(bvn_derivs, sig_sf_vec[k], I, true)
    #
    #     SensitiveFloats.clear!(cfg.log_pdf)
    #     SensitiveFloats.clear!(cfg.pdf)
    #
    #     # This is redundant, but it's what eval_bvn_pdf returns.
    #     log_pdf.v[] = log(bvn_derivs.f_pre[1])
    #
    #     if calculate_gradient
    #         for ind=1:2
    #             log_pdf.d[psf_ids.mu[ind]] = bvn_derivs.bvn_u_d[ind]
    #         end
    #         for ind=1:3
    #             log_pdf.d[sigma_ids[ind]] = bvn_derivs.bvn_s_d[ind]
    #         end
    #         log_pdf.d[psf_ids.weight] = 0
    #
    #         for ind1 = 1:2, ind2 = 1:2
    #             log_pdf.h[psf_ids.mu[ind1], psf_ids.mu[ind2]] =
    #                 bvn_derivs.bvn_uu_h[ind1, ind2]
    #         end
    #         for mu_ind = 1:2, sig_ind = 1:3
    #             log_pdf.h[psf_ids.mu[mu_ind], sigma_ids[sig_ind]] =
    #             log_pdf.h[sigma_ids[sig_ind], psf_ids.mu[mu_ind]] =
    #                 bvn_derivs.bvn_us_h[mu_ind, sig_ind]
    #         end
    #         for ind1 = 1:3, ind2 = 1:3
    #             log_pdf.h[sigma_ids[ind1], sigma_ids[ind2]] =
    #                 bvn_derivs.bvn_ss_h[ind1, ind2]
    #         end
    #     end
    #
    #     pdf_val = exp(log_pdf.v[])
    #     pdf.v[] = pdf_val
    #
    #     if calculate_gradient
    #         for ind1 = 1:length(PsfParams)
    #             if ind1 == psf_ids.weight
    #                 pdf.d[ind1] = pdf_val
    #             else
    #                 pdf.d[ind1] = psf_params[k][psf_ids.weight] * pdf_val * log_pdf.d[ind1]
    #             end
    #
    #             for ind2 = 1:ind1
    #                 pdf.h[ind1, ind2] = pdf.h[ind2, ind1] =
    #                     psf_params[k][psf_ids.weight] * pdf_val *
    #                     (log_pdf.h[ind1, ind2] + log_pdf.d[ind1] * log_pdf.d[ind2])
    #             end
    #         end
    #
    #         # Weight hessian terms.
    #         for ind1 = 1:length(PsfParams)
    #             pdf.h[psf_ids.weight, ind1] = pdf.h[ind1, psf_ids.weight] =
    #                 pdf_val * log_pdf.d[ind1]
    #         end
    #
    #     end
    #
    #     pdf.v[] *= psf_params[k][psf_ids.weight]
    #
    #     SensitiveFloats.add_sources_sf!(pixel_value, pdf, k)
    # end
    return nothing
end

##################################
# Callable Types Passed to Optim #
##################################

# Objective #
#-----------#

immutable Objective{N,T} <: Function
    raw_psf::Matrix{T}
    cfg::Config{N,T}
end

function (f::Objective)(x::Vector)
    from_vector!(f.cfg.free_params, x)
    to_bound!(f.cfg.bound_params, f.cfg.free_params, f.cfg.constraints)
    evaluate_psf_fit!(f.cfg, f.raw_psf, false)
    return f.cfg.sf_free.v[]
end

# Gradient #
#----------#

immutable Gradient{N,T} <: Function
    raw_psf::Matrix{T}
    cfg::Config{N,T}
end

"""
Computes both the gradient the objective function's value
"""
function (f::Gradient)(x::Vector, result::Vector)
    # TODO
end

# HessianVectorProduct #
#----------------------#

immutable HessianVectorProduct{N,T} <: Function
    raw_psf::Matrix{T}
    cfg::Config{N,T}
end

"""
Computes hessian-vector products
"""
function (f::HessianVectorProduct)(x::Vector, v::Vector, result::Vector)
    # TODO
    return nothing
end

############
# fit_psf! #
############

function fit_psf!(raw_psf::Matrix{Float64},
                  init_params::ComponentParams{Float64},
                  cfg::Config = Config(raw_psf, params))
    enforce_references!(raw_psf, init_params, cfg)
    enforce!(cfg.bound_params, cfg.constraints)
    to_free!(cfg.free_params, cfg.bound_params, cfg.constraints)
    x = to_vector(cfg.free_params)
    ddf = TwiceDifferentiableHV(Objective(raw_psf, coordinates, cfg),
                                Gradient(raw_psf, coordinates, cfg),
                                HessianVectorProduct(raw_psf, coordinates, cfg)),
    R = Optim.MultivariateOptimizationResults{Float64,1,CGTrustRegion{Float64}}
    return Optim.optimize(ddf, x, cfg.trust_region, cfg.optim_options)::R
end

end # module
