using ..Optimization: MinConfig, ParameterConstraint, BoxConstraint,
                      SimplexConstraint, ConstraintBatch

immutable PSFBundle{T}
    bvn_derivs::BivariateNormalDerivatives{T}
    coordinates::Vector{SVector{2,T}}
    log_pdf::SensitiveFloat{T}
    pdf::SensitiveFloat{T}
    pixel_value::SensitiveFloat{T}
    squared_error::SensitiveFloat{T}
    function (::Type{PSFBundle{T}}){T}(raw_psf::Matrix{T}, K::Int, constraints::ConstraintBatch)
        bvn_derivs = BivariateNormalDerivatives{T}()
        log_pdf = SensitiveFloat{T}(length(PsfParams), 1, true)
        pdf = SensitiveFloat{T}(length(PsfParams), 1, true)
        pixel_value = SensitiveFloat{T}(length(PsfParams), K, true)
        squared_error = SensitiveFloat{T}(length(PsfParams), K, true)
        return new{T}(bvn_derivs, coordinates, log_pdf, pdf, pixel_value, squared_error)
    end
end

immutable PSFFitConfig{T,N}
    psf_bundle::PSFBundle{T}
    min_cfg::MinConfig{T,N}
end

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
    return Optim.NewtonTrustRegion(; initial_delta=initial_delta, delta_hat=delta_hat,
                                   eta=eta, rho_lower=rho_lower, rho_upper=rho_upper)
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
