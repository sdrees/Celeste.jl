"""
Convolve the current locations and galaxy shapes with the PSF.  If
calculate_gradient is true, also calculate derivatives for
active sources.

Args:
 - S (formerly from ElboArgs)
 - patches (formerly from ElboArgs)
 - vp: (formerly from ElboArgs)
 - active_sources (formerly from ElboArgs)
 - psf_K: Number of psf components (psf from patches object)
 - n: the image id (not the band)
 - calculate_gradient: Whether to calculate derivatives for active sources.

Returns:
 - star_mcs: An array of BvnComponents with indices
    - PSF component
    - Source (index within active_sources)
 - gal_mcs: An array of BvnComponents with indices
    - PSF component
    - Galaxy component
    - Galaxy type
    - Source (index within active_sources)
  Hessians are only populated for s in ea.active_sources.
"""
function load_bvn_mixtures{NumType <: Number}(
                    S::Int64,
                    patches::Matrix{SkyPatch},
                    source_params::Vector{Vector{NumType}},
                    active_sources::Vector{Int},
                    psf_K::Int64,
                    n::Int;
                    calculate_gradient::Bool=true)
    # TODO: do not keep any derviative information if the sources are not in
    # active_sources.
    star_mcs = Matrix{BvnComponent{NumType}}(psf_K, S)
    gal_mcs  = Array{GalaxyCacheComponent{NumType}}(psf_K, 8, 2, S)

    for s in 1:S
        psf = patches[s, n].psf
        sp  = source_params[s]

        # TODO: it's a lucky coincidence that lidx.u = ids.u.
        # That's why this works when called from both log_prob.jl with `sp`
        # and elbo_objective.jl with `vp`.
        # We need a safer way to let both methods call this method.
        world_loc = sp[lidx.u]
        m_pos = Model.linear_world_to_pix(patches[s, n].wcs_jacobian,
                                          patches[s, n].center,
                                          patches[s, n].pixel_center, world_loc)

        # Convolve the star locations with the PSF.
        for k in 1:psf_K
            pc = psf[k]
            mean_s = @SVector NumType[pc.xiBar[1] + m_pos[1], pc.xiBar[2] + m_pos[2]]
            star_mcs[k, s] =
              BvnComponent(mean_s, pc.tauBar, pc.alphaBar, false)
        end

        # Convolve the galaxy representations with the PSF.
        for i = 1:2 # i indexes dev vs exp galaxy types.
            e_dev_dir = (i == 1) ? 1. : -1.
            e_dev_i = (i == 1) ? sp[lidx.e_dev] : 1. - sp[lidx.e_dev]

            # Galaxies of type 1 have 8 components, and type 2 have 6 components.
            for j in 1:[8,6][i]
                for k = 1:psf_K
                    gal_mcs[k, j, i, s] = GalaxyCacheComponent(
                        e_dev_dir, e_dev_i, galaxy_prototypes[i][j], psf[k],
                        m_pos,
                        sp[lidx.e_axis], sp[lidx.e_angle], sp[lidx.e_scale],
                        calculate_gradient && (s in active_sources))
                end
            end
        end
    end

    star_mcs, gal_mcs
end


"""
Populate fs0m and fs1m for source s in the a given pixel.

Non-standard args:
    - x: The pixel location in the image
"""
function populate_gal_fsm!{NumType <: Number}(
                    bvn_derivs::BivariateNormalDerivatives{NumType},
                    fs1m::SensitiveFloat{NumType},
                    s::Int,
                    x::SVector{2,Float64},
                    is_active_source::Bool,
                    wcs_jacobian::Matrix{Float64},
                    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4};
                    num_allowed_sd=Inf)
    clear!(fs1m)
    for i = 1:2 # Galaxy types
        for j in 1:8 # Galaxy component
            # If i == 2 then there are only six galaxy components.
            if (i == 1) || (j <= 6)
                for k = 1:size(gal_mcs, 1) # PSF component
                    if (num_allowed_sd == Inf ||
                        check_point_close_to_bvn(
                            gal_mcs[k, j, i, s].bmc, x, num_allowed_sd))
                        accum_galaxy_pos!(
                            bvn_derivs, fs1m,
                            gal_mcs[k, j, i, s], x, wcs_jacobian,
                            is_active_source)
                    end
                end
            end
        end
    end
end


"""
Populate fs0m and fs1m for source s in the a given pixel.

Non-standard args:
    - x: The pixel location in the image
"""
function populate_fsm!{NumType <: Number}(
                    bvn_derivs::BivariateNormalDerivatives{NumType},
                    fs0m::SensitiveFloat{NumType},
                    fs1m::SensitiveFloat{NumType},
                    s::Int,
                    x::SVector{2,Float64},
                    is_active_source::Bool,
                    wcs_jacobian::Matrix{Float64},
                    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
                    star_mcs::Array{BvnComponent{NumType}, 2};
                    num_allowed_sd=Inf)
    clear!(fs0m)
    for k = 1:size(star_mcs, 1) # PSF component
        if (num_allowed_sd == Inf ||
            check_point_close_to_bvn(star_mcs[k, s], x, num_allowed_sd))
            accum_star_pos!(
                bvn_derivs, fs0m,
                star_mcs[k, s], x, wcs_jacobian, is_active_source)
        end
    end

    populate_gal_fsm!(bvn_derivs, fs1m,
                      s, x, is_active_source,
                      wcs_jacobian, gal_mcs;
                      num_allowed_sd=num_allowed_sd)
end


"""
Add the contributions of a star's bivariate normal term to the ELBO,
by updating elbo_vars.fs0m in place.

Args:
    - bvn_derivs: (formerly from elbo_vars)
    - fs0m: sensitive floats populated by this method
    - calculate_gradient: the and of is_active_source and formerly elbo_vars.calculate_gradient
    - s: The index of the current source in 1:S
    - bmc: The component to be added
    - x: An offset for the component in pixel coordinates (e.g. a pixel location)
    - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.
"""
function accum_star_pos!{NumType <: Number}(
                    bvn_derivs::BivariateNormalDerivatives{NumType},
                    fs0m::SensitiveFloat{NumType},
                    bmc::BvnComponent{NumType},
                    x::SVector{2,Float64},
                    wcs_jacobian::Array{Float64, 2},
                    is_active_source::Bool)
    eval_bvn_pdf!(bvn_derivs, bmc, x)

    if fs0m.has_gradient && is_active_source
        get_bvn_derivs!(bvn_derivs, bmc, true, false)
    end

    fs0m.v[] += bvn_derivs.f_pre[1]

    if fs0m.has_gradient && is_active_source
        transform_bvn_ux_derivs!(bvn_derivs, wcs_jacobian)
        bvn_u_d = bvn_derivs.bvn_u_d
        bvn_uu_h = bvn_derivs.bvn_uu_h

        # Accumulate the derivatives.
        for u_id in 1:2
            fs0m.d[star_ids.u[u_id]] += bvn_derivs.f_pre[1] * bvn_u_d[u_id]
        end
    end
end


"""
Add the contributions of a galaxy component term to the ELBO by
updating fs1m in place.
"""
function accum_galaxy_pos!{NumType <: Number}(
                    bvn_derivs::BivariateNormalDerivatives{NumType},
                    fs1m::SensitiveFloat{NumType},
                    gcc::GalaxyCacheComponent{NumType},
                    x::SVector{2,Float64},
                    wcs_jacobian::Array{Float64, 2},
                    is_active_source::Bool)
    eval_bvn_pdf!(bvn_derivs, gcc.bmc, x)
    f = bvn_derivs.f_pre[1] * gcc.e_dev_i
    fs1m.v[] += f

    if fs1m.has_gradient && is_active_source
        get_bvn_derivs!(bvn_derivs, gcc.bmc, fs1m.has_gradient)
        transform_bvn_derivs!(bvn_derivs, gcc.sig_sf, wcs_jacobian)

        bvn_u_d = bvn_derivs.bvn_u_d
        bvn_s_d = bvn_derivs.bvn_s_d

        # Accumulate the derivatives.
        for u_id in 1:2
            fs1m.d[gal_ids.u[u_id]] += f * bvn_u_d[u_id]
        end

        for gal_id in 1:length(gal_shape_ids)
            fs1m.d[gal_shape_alignment[gal_id]] += f * bvn_s_d[gal_id]
        end

        # The e_dev derivative. e_dev just scales the entire component.
        # The direction is positive or negative depending on whether this
        # is an exp or dev component.
        fs1m.d[gal_ids.e_dev] += gcc.e_dev_dir * bvn_derivs.f_pre[1]

    end # if is_active_source
end
