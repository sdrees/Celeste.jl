#########
# Types #
#########

# BVNComponent #
#--------------#

immutable BVNComponent{T}
    z::T
    major_σ::T
    mean::SVector{2,T}
    precision::SMatrix{2,2,T,4}
end

function BVNComponent{T}(mean::SVector{2,T}, cov::SMatrix{2,2,T,4}, weight::T)
    return BVNComponent{T}(1 / (sqrt(det(cov)) * 2π) * weight,
                           sqrt(max(cov[1, 1], cov[2, 2]))
                           mean,
                           inv(cov))
end

# BVNDerivatives #
#----------------#

immutable BVNDerivatives{T}
    # Pre-allocated memory for py1, py2, and f when evaluating BVNs
    py1::RefValue{T}
    py2::RefValue{T}
    f_pre::RefValue{T}

    # Derivatives of a bvn with respect to (x, σ).
    bvn_x_d::Vector{T}
    bvn_σ_d::Vector{T}

    # intermediate values used in d bvn / d(x, σ)
    dpy1_dσ::Vector{T}
    dpy2_dσ::Vector{T}

    # Derivatives of a bvn with respect to (u, shape)
    bvn_u_d::Vector{T}
    bvn_s_d::Vector{T}

    function (::Type{BivariateNormalDerivatives{T}}){T}()
        py1 = RefValue{T}()
        py2 = RefValue{T}()
        f_pre = RefValue{T}()
        bvn_x_d = zeros(T, 2)
        bvn_σ_d = zeros(T, 3)
        dpy1_dσ = zeros(T, 3)
        dpy2_dσ = zeros(T, 3)
        bvn_u_d = zeros(T, 2) # derivatives wrt u
        bvn_s_d = zeros(T, length(gal_shape_ids)) # shape deriviatives
        new{T}(py1, py2, f_pre,
               bvn_x_d, bvn_σ_d,
               dpy1_dσ, dpy2_dσ,
               bvn_u_d, bvn_s_d)
    end
end

# GalaxyCacheComponent #
#----------------------#

immutable GalaxyCacheComponent{T}
    e_dev_dir::Float64
    e_dev_i::T
    bc::BVNComponent{T}
    σ_derivs::Matrix{T}
end

function GalaxyCacheComponent{T}(e_dev_dir::Float64, e_dev_i::T,
                                 gc::GalaxyComponent, pc::PSFComponent,
                                 u::Vector{T}, e_axis::T, e_angle::T, e_scale::T,
                                 calculate_gradient::Bool)
    bc = BVNComponent(SVector(pc.xiBar[1] + u[1], pc.xiBar[2] + u[2]),
                      pc.tauBar + gc.nuBar * get_bvn_cov(e_axis, e_angle, e_scale),
                      pc.alphaBar * gc.etaBar)
    if calculate_gradient
        σ_derivs = Matrix{T}(3, length(gal_shape_ids))
        sig_sf = GalaxySigmaDerivs(e_angle, e_axis, e_scale, XiXi, calculate_hessian)
        scale!(σ_derivs, gc.nuBar)
    else
        σ_derivs = Matrix{T}(0, 0)
    end
    GalaxyCacheComponent(e_dev_dir, e_dev_i, bc, σ_derivs)
end

########################
# Zero-Order Functions #
########################

function bvn_cov{T}(ab::T, angle::T, scale::T)
    cp = cos(angle)
    sp = sin(angle)
    ab_term = (ab ^ 2 - 1)
    scale_squared = scale ^ 2
    off_diag_term = -scale_squared * cp * sp * ab_term
    diag_term_1 = scale_squared * (1 + ab_term * (sp ^ 2))
    diag_term_2 = scale_squared * (1 + ab_term * (cp ^ 2))
    @SMatrix T[diag_term_1    off_diag_term;
               off_diag_term     diag_term_2]
end

function check_point_close_to_bvn(bc::BVNComponent, x::SVector{2}, allowed_σ)
    return sqrt(norm(x - bc.mean)) < (allowed_σ * bc.major_σ)
end

####################################
# Derivative-Calculating Functions #
####################################

function eval_bvn_pdf!(derivs::BVNDerivatives,
                       bc::BVNComponent,
                       x::SVector{2})
    c1 = x[1] - bc.mean[1]
    c2 = x[2] - bc.mean[2]
    py1 = bc.precision[1, 1] * c1 + bc.precision[1, 2] * c2
    py2 = bc.precision[2, 1] * c1 + bc.precision[2, 2] * c2
    derivs.py1[] = py1
    derivs.py2[] = py2
    derivs.f_pre[] = bc.z * exp(-0.5 * (c1 * py1 + c2 * py2))
    return nothing
end

function update_bvn_derivs!(derivs::BVNDerivatives, bc::BVNComponent{NumType})
    py1 = derivs.py1[]
    py2 = derivs.py2[]

    bvn_x_d = derivs.bvn_x_d
    @inbounds bvn_x_d[1] = -py1
    @inbounds bvn_x_d[2] = -py2

    bvn_sig_d = bvn_derivs.bvn_sig_d
    @inbounds bvn_sig_d[1] = 0.5 * (py1 * py1 - bc.precision[1, 1])
    @inbounds bvn_sig_d[2] = (py1 * py2) - bc.precision[1, 2]
    @inbounds bvn_sig_d[3] = 0.5 * (py2 * py2 - bvn.precision[2, 2])
    return nothing
end

function galaxy_σ_derivs!{T}(result::Matrix{T}, e_angle::T, e_axis::T, e_scale::T, XiXi::SMatrix{2,2,T,4})
    @assert size(result) == (3, length(gal_shape_ids))

    cos_sin = cos(e_angle)sin(e_angle)
    sin_sq = sin(e_angle)^2
    cos_sq = cos(e_angle)^2

    e_axis_partial_coeff = 2 * e_axis * e_scale * e_scale
    e_axis_partial_1 = e_axis_partial_coeff * sin_sq
    e_axis_partial_2 = e_axis_partial_coeff * -cos_sin
    e_axis_partial_3 = e_axis_partial_coeff * cos_sq
    for i in gal_shape_ids.e_axis
        @inbounds result[1, i] = e_axis_partial_1
        @inbounds result[2, i] = e_axis_partial_2
        @inbounds result[3, i] = e_axis_partial_3
    end

    e_angle_partial_coeff = e_scale * e_scale * (e_axis * e_axis - 1)
    e_angle_partial_1 = e_angle_partial_coeff * (cos_sin + cos_sin)
    e_angle_partial_2 = e_angle_partial_coeff * (sin_sq - cos_sq)
    e_angle_partial_3 = e_angle_partial_coeff * -(cos_sin + cos_sin)
    for i in gal_shape_ids.e_angle
        @inbounds result[1, i] = e_angle_partial_1
        @inbounds result[2, i] = e_angle_partial_2
        @inbounds result[3, i] = e_angle_partial_3
    end

    e_scale_partial_coeff = 2.0 / e_scale
    e_scale_partial_1 = e_scale_partial_coeff * XiXi[1]
    e_scale_partial_2 = e_scale_partial_coeff * XiXi[2]
    e_scale_partial_3 = e_scale_partial_coeff * XiXi[4]
    for i in gal_shape_ids.e_scale
        @inbounds result[1, i] = e_scale_partial_1
        @inbounds result[2, i] = e_scale_partial_2
        @inbounds result[3, i] = e_scale_partial_3
    end

    return nothing
end

############################
# Transformation Functions #
############################

function transform_bvn_ux_derivs!(derivs::BVNDerivatives, wcs_jacobian)
    # Note that dxA_duB = -wcs_jacobian[A, B].  (It is minus the jacobian
    # because the object position affects the bvn.mean term, which is
    # subtracted from the pixel location as defined in bvn_sf.d.)
    bvn_u_d = derivs.bvn_u_d
    bvn_x_d = derivs.bvn_x_d
    @inbounds bvn_u_d[1] = -(bvn_x_d[1] * wcs_jacobian[1, 1] + bvn_x_d[2] * wcs_jacobian[2, 1])
    @inbounds bvn_u_d[2] = -(bvn_x_d[1] * wcs_jacobian[1, 2] + bvn_x_d[2] * wcs_jacobian[2, 2])
    return nothing
end

function transform_bvn_derivs!(derivs::BVNDerivatives, galaxy_σ_derivs, wcs_jacobian)
    transform_bvn_ux_derivs!(bvn_derivs, wcs_jacobian)
    bvn_s_d = bvn_derivs.bvn_s_d
    bvn_sig_d = bvn_derivs.bvn_sig_d
    for shape_id in 1:length(gal_shape_ids)
        @inbounds bvn_s_d[shape_id] = (bvn_sig_d[1] * galaxy_σ_derivs[1, shape_id] +
                                       bvn_sig_d[2] * galaxy_σ_derivs[2, shape_id] +
                                       bvn_sig_d[3] * galaxy_σ_derivs[3, shape_id])
    end
    return nothing
end
