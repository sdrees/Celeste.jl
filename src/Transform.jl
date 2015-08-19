# Convert between different parameterizations.

module Transform

using Celeste
using CelesteTypes

import Util
VERSION < v"0.4.0-dev" && using Docile
@docstrings

export DataTransform, ParamBounds, get_mp_transform

# The box bounds for a symbol.
typealias ParamBounds Dict{Symbol, (Union(Float64, Vector{Float64}), Union(Float64, Vector{Float64})) }

#####################
# Conversion to and from vectors.

function free_vp_to_vector{NumType <: Number}(vp::Vector{NumType},
                                              omitted_ids::Vector{Int64})
    # vp = variational parameters
    # omitted_ids = ids in ParamIndex
    #
    # There is probably no use for this function, since you'll only be passing
    # trasformations to the optimizer, but I'll include it for completeness.

    left_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    new_p = 1:length(left_ids)
    new_p[:] = vp[left_ids]
end


function vector_to_free_vp!{NumType <: Number}(xs::Vector{NumType},
                                               vp_free::Vector{NumType},
                                               omitted_ids::Vector{Int64})
    # xs: A vector created from free variational parameters.
    # free_vp: Free variational parameters.  Only the ids not in omitted_ids
    #   will be updated.
    # omitted_ids: Ids to omit (from ids_free)

    left_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    vp_free[left_ids] = xs
end


###############################################
# Functions for a "free transform".

function unbox_parameter{NumType <: Number}(
  param::Union(NumType, Vector{NumType}), upper_bound::Float64, lower_bound::Float64)
    @assert(all(lower_bound .< param .< upper_bound),
            "param outside bounds: $param ($lower_bound, $upper_bound)")
    param_scaled = (param - lower_bound) / (upper_bound - lower_bound)
    Util.inv_logit(param_scaled)
end

function box_parameter{NumType <: Number}(
  free_param::Union(NumType, Vector{NumType}), upper_bound::Float64, lower_bound::Float64)
    Util.logit(free_params) * (upper_bound - lower_bound) + lower_bound
end

@doc """
Updates free_deriv in place.  <param> is the parameter that lies
within the box constrains, and <deriv> is the derivative with respect
to these paraemters.
""" ->
function unbox_derivative{NumType <: Number}(
  param::Union(NumType, Vector{NumType}), deriv::Union(NumType, Vector{NumType}),
  upper_bound::Float64, lower_bound::Float64)
    @assert(length(param) == length(deriv) == length(free_deriv),
            "Wrong length parameters for unbox_sensitive_float")

    # Box constraints.  Strict inequality is not required for derivatives.
    @assert(all(lower_bound .<= param .<= upper_bound),
            "param outside bounds: $param ($lower_bound, $upper_bound)")
    param_scaled = (param - lower_bound) ./ (upper_bound - lower_bound)

    deriv .* param_scaled .* (1 - param_scaled) .* (upper_bound - lower_bound)
end


@doc """
Convert a variational parameter vector to an unconstrained version using
the lower bounds lbs and ubs (which are expressed)
""" ->
function vp_to_free!{NumType <: Number}(
  vp::Vector{NumType}, vp_free::Vector{NumType}, bounds::ParamBounds)
    # Simplicial constriants.

    # The original script used "a" to only
    # refer to the probability of being a galaxy, which is now the
    # second component of a.
    vp_free[ids_free.a[1]] = Util.inv_logit(vp[ids.a[2]])

    # In contrast, the original script used the last component of k
    # as the free parameter.
    vp_free[ids_free.k[1, :]] = Util.inv_logit(vp[ids.k[1, :]])

    # Box constraints.
    for (param, limits) in bounds
        vp_free[ids_free.(param)] =
          unbox_parameter(vp[ids.(param)], limits[1], limits[2])
    end
end


function free_to_vp!{NumType <: Number}(
  vp_free::Vector{NumType}, vp::Vector{NumType}, bounds::ParamBounds)
    # Convert an unconstrained to an constrained variational parameterization.

    # Simplicial constriants.
    vp[ids.a[2]] = Util.logit(vp_free[ids_free.a[1]])
    vp[ids.a[1]] = 1.0 - vp[ids.a[2]]

    vp[s][ids.k[1, :]] = Util.logit(vp_free[s][ids_free.k[1, :]])
    vp[s][ids.k[2, :]] = 1.0 - vp[s][ids.k[1, :]]

    # Box constraints.
    for (param, limits) in bounds
        vp[ids.(param)] =
          box_parameter(vp_free[ids_free.(param)], limits[1], limits[2])
    end
end


@doc """
Return the derviatives with respect to the unboxed
parameters given derivatives with respect to the boxed parameters.
""" ->
function unbox_param_derivative{NumType <: Number}(
  vp::Vector{NumType}, d::Vector{NumType}, bounds::ParamBounds)

  d_free = zeros(NumType, length(UnconstrainedParams))

  # TODO: write in general form.  Note that the old "a" is now a[2].
  # Simplicial constriants.
  this_a = vp[ids.a[2]]
  d_free[ids_free.a[1]] =
      (d[ids.a[2]] - d[ids.a[1]]) * this_a * (1.0 - this_a)

  this_k = collect(vp[ids.k[1, :]])
  d_free[collect(ids_free.k[1, :])] =
      (d[collect(ids.k[1, :])] - d[collect(ids.k[2, :])]) .* this_k .* (1.0 - this_k)

  for (param, limits) in bounds
      d_free[ids_free.(param)] =
        unbox_derivative(vp[ids.(param)], d[ids.(param)], limits[1], limits[2])
  end

  d_free
end


#########################
# Define the exported variables.


@doc """
Functions to move between a single source's variational parameters and a
transformation of the data for optimization.

to_vp: A function that takes transformed parameters and returns variational parameters
from_vp: A function that takes variational parameters and returned transformed parameters
to_vp!: A function that takes (transformed paramters, variational parameters) and updates
  the variational parameters in place
from_vp!: A function that takes (variational paramters, transformed parameters) and updates
  the transformed parameters in place
...
transform_sensitive_float: A function that takes (sensitive float, model parameters)
  where the sensitive float contains partial derivatives with respect to the
  variational parameters and returns a sensitive float with total derivatives with
  respect to the transformed parameters. """ ->
type DataTransform
	to_vp::Function
	from_vp::Function
	to_vp!::Function
	from_vp!::Function
  vp_to_vector::Function
  vector_to_vp!::Function
	transform_sensitive_float::Function
  bounds::Vector{ParamBounds}
end

DataTransform(bounds::Vector{ParamBounds}) = begin

  function from_vp!{NumType <: Number}(
    vp::VariationalParams{NumType}, vp_free::VariationalParams{NumType})
      S = length(vp)
      @assert S == length(bounds)
      for s=1:S
        vp_to_free!(vp[s], vp_free[s], bounds[s])
      end
  end

  function from_vp{NumType <: Number}(vp::VariationalParams{NumType})
      vp_free = [ zeros(NumType, length(ids_free)) for s = 1:length(vp)]
      from_vp!(vp, vp_free)
      vp_free
  end

  function to_vp!{NumType <: Number}(
    vp_free::FreeVariationalParams{NumType}, vp::VariationalParams{NumType})
      S = length(vp_free)
      @assert S == length(bounds)
      for s=1:S
        free_to_vp!(vp_free[s], vp[s], bounds[s])
      end
  end

  function to_vp{NumType <: Number}(vp_free::FreeVariationalParams{NumType})
      vp = [ zeros(length(CanonicalParams)) for s = 1:length(vp_free)]
      to_vp!(vp_free, vp)
      vp
  end

  function vp_to_vector{NumType <: Number}(vp::VariationalParams{NumType},
                                           omitted_ids::Vector{Int64})
      vp_trans = from_vp(vp)
      trans_vp_to_vector(vp_trans, omitted_ids)
  end

  function vector_to_vp!{NumType <: Number}(xs::Vector{NumType},
                                            vp::VariationalParams{NumType},
                                            omitted_ids::Vector{Int64})
      # This needs to update vp in place so that variables in omitted_ids
      # stay at their original values.
      vp_trans = from_vp(vp)
      vector_to_trans_vp!(xs, vp_trans, omitted_ids)
      to_vp!(vp_trans, vp)
  end

  # Given a sensitive float with derivatives with respect to all the
  # constrained parameters, calculate derivatives with respect to
  # the unconstrained parameters.
  #
  # Note that all the other functions in ElboDeriv calculated derivatives with
  # respect to the unconstrained parameterization.
  function transform_sensitive_float{NumType <: Number}(
    sf::SensitiveFloat, mp::ModelParams{NumType})

      # Require that the input have all derivatives defined.
      @assert size(sf.d) == (length(CanonicalParams), mp.S) == length(bounds)

      sf_free = zero_sensitive_float(UnconstrainedParams, NumType, mp.S)
      sf_free.v = sf.v

      for s in 1:mp.S
        sf_free.d[:, s] =
          unbox_variational_params(mp.vp[s], sf.d[:, s][:], bounds[s])
      end

      sf_free
  end

  DataTransform(to_vp, from_vp, to_vp!, from_vp!, vp_to_vector, vector_to_vp!,
                transform_sensitive_float, bounds)
end

function get_mp_transform(mp::ModelParams; loc_width::Float64=1e-3)
  bounds = Array(ParamBounds, 3)
  for s=1:mp.S
    bounds[s] = ParamBounds()
    bounds[s][:u] = (mp.vp[s][ids.u] - loc_width, mp.vp[s][ids.u] + loc_width)
    bounds[s][:r1] = (1e-4, 1e12)
    bounds[s][:r2] = (1e-4, 0.1)
    bounds[s][:c1] = (-10., 10.)
    bounds[s][:c2] = (1e-4, 1.)
    bounds[s][:e_dev] = (1e-2, 1 - 1e-2)
    bounds[s][:e_axis] = (1e-4, 1 - 1e-4)
    bounds[s][:e_angle] = (-1e10, 1e10)
    bounds[s][:e_scale] = (0.2, 15.)
  end
  DataTransform(bounds)
end


end
