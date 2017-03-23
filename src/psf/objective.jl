using ..Optimization: MinConfig, ParameterConstraint, BoxConstraint,
                      SimplexConstraint, ConstraintBatch

function psf_constraints{T}(params::ComponentParams{T})
    n_components = length(params)
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

"""
evaluate_psf_fit but with pre-allocated memory for intermediate calculations.
"""
function evaluate_psf_fit!{NumType <: Number}(
        psf_params::Vector{Vector{NumType}}, raw_psf::Matrix{Float64},
        x_mat::Matrix{SVector{2,Float64}},
        bvn_derivs::BivariateNormalDerivatives{NumType},
        log_pdf::SensitiveFloat{NumType},
        pdf::SensitiveFloat{NumType},
        pixel_value::SensitiveFloat{NumType},
        squared_error::SensitiveFloat{NumType},
        calculate_gradient::Bool)
    K = length(psf_params)
    sigma_vec, sig_sf_vec, bvn_vec = get_sigma_from_params(psf_params)
    clear!(squared_error)

    @inbounds for x_ind in 1:length(x_mat)
        clear!(pixel_value)
        evaluate_psf_pixel_fit!(
                x_mat[x_ind], psf_params, sigma_vec, sig_sf_vec, bvn_vec,
                bvn_derivs, log_pdf, pdf, pixel_value, calculate_gradient)

        diff = (pixel_value.v[] - raw_psf[x_ind])
        squared_error.v[] +=    diff ^ 2
        if calculate_gradient
            for ind1 = 1:length(squared_error.d)
                squared_error.d[ind1] += 2 * diff * pixel_value.d[ind1]
                for ind2 = 1:ind1
                    squared_error.h[ind1, ind2] +=
                        2 * (diff * pixel_value.h[ind1, ind2] +
                                 pixel_value.d[ind1] * pixel_value.d[ind2]')
                    squared_error.h[ind2, ind1] = squared_error.h[ind1, ind2]
                end
            end
        end # if calculate_gradient
    end

    squared_error
end
