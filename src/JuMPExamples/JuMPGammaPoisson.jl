using JuMP
using Distributions
using Gadfly



N = 5000
S = 5
point_spread = (0.5 / S) / 5

#################################
# Generate the celestial objects
mu = linspace(0.25, 0.75, S)

star_prob = 0.5
prior_upsilon = [1.0, 10.0]
prior_phi = [0.5, 0.5]

star_brightness = rand(Gamma(prior_upsilon[1], prior_phi[1]), S)
galaxy_brightness = rand(Gamma(prior_upsilon[2], prior_phi[2]), S)
true_is_star = rand(Bernoulli(star_prob), S)
true_brightness = true_is_star .* star_brightness + (1 - true_is_star) .* galaxy_brightness

# Generate the readings
m_loc = linspace(0, 1, N)
# Why is this convert statement necessary?
phi_ns = convert(Array{Float64, 2}, [pdf(Normal(0, point_spread), m_i - mu_s) for m_i in m_loc, mu_s in mu])
x = [convert(Float64, rand(Poisson(b))) for b in phi_ns * true_brightness]

plot(x=m_loc, y=x)

#########################
# Optimize with JuMP

# Unconstrain the variables.  Does this really help at all?
@defNLExpr(q_chi[s=1:S], exp(logit_chi[s]) / (1 + exp(logit_chi[s])))
@defNLExpr(q_gamma[s=1:S, a=1:2], exp(log_gamma[s, a]))
@defNLExpr(q_zeta[s=1:S, a=1:2], exp(log_zeta[s, a]))

# Define the a expectations.
@defNLExpr(e_a[s=1:S, 1], q_chi[s])
@defNLExpr(e_a[s=1:S, 2], 1 - q_chi[s])

# Define the r expectations.
@defNLExpr(e_ra[s=1:S, a=1:2],     q_gamma[s, a] * q_zeta[s, a])
@defNLExpr(e_ra2[s=1:S, a=1:2],    (1 + q_gamma[s, a]) * q_gamma[s, a] * (q_zeta[s, a] ^ 2))
@defNLExpr(e_log_ra[s=1:S, a=1:2], digamma(q_gamma[s, a]) + log(q_zeta[s, a]))
@defNLExpr(e_r[s=1:S],             sum{e_a[s, a] * e_ra[s, a], a=1:2})
@defNLExpr(var_r[s=1:S],           sum{e_a[s, a] * e_ra2[s, a], a=1:2} - (e_r[s]) ^ 2)

# Define the F expectations.
@defNLExpr(e_fns[n=1:N, s=1:S], e_r[s] * phi_ns[n, s])
@defNLExpr(var_fns[n=1:N, s=1:S], var_r[s] * (phi_ns[n, s])^2)
@defNLExpr(e_fn[n=1:N], sum{e_fns[n, s], s=1:S})
@defNLExpr(var_fn[n=1:N], sum{var_fns[n, s], s=1:S})
@defNLExpr(e_log_fn[n=1:N], log(e_fn[n]) - var_fn[n] / (2 * e_fn[n] ^ 2))
@defNLExpr(e_log_lik, sum{x[n] * e_log_fn[n] - e_fn[n], n=1:N})

# Define the entropy.
@defNLExpr(ent_rsa[s=1:S, a=1:2],
	       q_gamma[s, a] + log(q_zeta[s, a]) + lgamma(q_zeta[s, a]) +
	       (1 - q_zeta[s, a]) * digamma(q_zeta[s, a]))
@defNLExpr(ent_as[s=1:S],
	       -1 * q_chi[s] * log(q_chi[s]) - (1 - q_chi[s]) * log(1 - q_chi[s]))
@defNLExpr(entropy, sum{ent_rsa[s, a], s=1:S, a=1:2} + sum{ent_as[s], s=1:S})

# Define the expected priors.
@defNLExpr(e_ra_prior[s=1:S, a=1:2],
	       (prior_upsilon[a] - 1) * e_log_ra[s, a] -
	       e_ra[s, a] / prior_phi[a] -
	       prior_upsilon[a] * log(prior_phi[a]) - lgamma(prior_upsilon[a]))
@defNLExpr(e_a_prior[s=1:S], q_chi[s] * log(star_prob) + (1 - q_chi[s]) * log(1 - star_prob))
@defNLExpr(priors, sum{e_a[s, a] * e_ra_prior[s, a], s=1:S, a=1:2} + sum{e_a_prior[s], s=1:S})


# Define the model using the above expressions.
m = Model()
log_gamma_start = repmat(log(prior_upsilon)', S)
log_zeta_start = repmat(log(prior_phi)', S)
@defVar(m, log_gamma[s=1:S, a=1:2], start=log_gamma_start[s, a])
@defVar(m, log_zeta[s=1:S, a=1:2], start=log_zeta_start[s, a])
@defVar(m, logit_chi[s=1:S], start=[0 for s=1:S][s])
@setNLObjective(m, Max, e_log_lik + entropy + priors)
solve(m)


# Check the model output.
m_log_gamma = getValue(log_gamma)
m_log_zeta = getValue(log_zeta)
m_logit_chi = getValue(logit_chi)

m_chi = [ exp(m_logit_chi[s]) / (1 + exp(m_logit_chi[s])) for s in 1:S]
m_brightness = [ exp(m_log_gamma[s, a]) * exp(m_log_zeta[s, a]) for s in 1:S, a in 1:2]

m_total_brightness = [ m_chi[s] * m_brightness[s, 1] + (1 - m_chi[s]) * m_brightness[s, 2] for s in 1:S]


######
# Just a hacky way to get the value of an NLExpr.

debug_model = Model()
epsilon = 1e-6
@defVar(debug_model, 5 <= debug_var <= 5 + epsilon)
@defNLExpr(log_debug_var, log(debug_var))
@setNLObjective(debug_model, Max, log_debug_var)
solve(debug_model)
getObjectiveValue(debug_model)

########

using JuMP
using ReverseDiffSparse
m = Model()
@defVar(m, 0 <= beta <= 1)
@defExpr(beta2, beta * beta)
setValue(beta, 2.0)
getValue(beta)
getValue(beta2)
@defNLExpr(beta2nl[s=1:5], s * beta * beta)
[ ReverseDiffSparse.getvalue(beta2nl[s], m.colVal) for s=1:5 ]


########

m = Model()
@defVar(m, baz)
@defNLExpr(foo[1], 5 * baz)
@defNLExpr(foo[2], -3 * baz)
@defNLExpr(bar, sum{foo[i], i=1:2})
setValue(baz, 2)
ReverseDiffSparse.getvalue(bar, m.colVal) # Expect 4, get -12

############

m = Model()
@defVar(m, baz)
@defExpr(foo[1], 5 * baz)
@defExpr(foo[2], -3 * baz)

########

m = Model()
@defVar(m, baz)
@defNLExpr(foo[s=1:2], (s == 1) * 5 * baz + (s == 2) * (-3 * baz))
@defNLExpr(bar, sum{foo[i], i=1:2})
setValue(baz, 2)
ReverseDiffSparse.getvalue(bar, m.colVal) # Get 4



##############
# Arrays of expressions

m = Model()
@defVar(m, bar)

foo = Array(Any, 5, 10)
for i = 1:5
	for j = 1:10
		foo[i, j] = @defNLExpr(placeholder, i * bar + j)
	end
end
setValue(bar, 2)

# This works:
foo_mat = [ ReverseDiffSparse.getvalue(foo[i, j], m.colVal) for i=1:5, j=1:10]

# The result of this is a parametric expression, not a value:
@defNLExpr(foo_element, foo[1, 1])
ReverseDiffSparse.getvalue(foo_element, m.colVal)

# The result of this is an error:
@defNLExpr(foo_sum, sum{foo[i, j], i=1:5, j=1:10})
ReverseDiffSparse.getvalue(foo_sum, m.colVal)
# ERROR: `+` has no method matching +(::Float64, ::ParametricExpression{0})
#  in _EXPRVAL_ at /home/rgiordan/.julia/v0.3/ReverseDiffSparse/src/revmode.jl:678
#  in getvalue at /home/rgiordan/.julia/v0.3/ReverseDiffSparse/src/revmode.jl:696


##############################
# Structures of expressions?  No.

m = Model()
@defVar(m, baz)
immutable MyThingy
	foo::Matrix{Any}
	bar::Any
	MyThingy(offset::Float64) = begin
		foo = Array(Any, 5, 10)
		for i = 1:5
			for j = 1:10
				foo[i, j] = @defNLExpr(placeholder, i * baz + j + offset)
			end
		end
		bar = @defNLExpr(baz ^ 2 + offset)
		new(foo, bar)
	end
end

thingy = MyThingy(-4.0)

@defNLExpr(thingy_sum, sum{thingy.foo[i, j], i=1:5, j=1:10});
setValue(baz, 2)
ReverseDiffSparse.getvalue(thingy_sum, m.colVal)



######################
m = Model()
@defVar(m, foo)
setValue(foo, 2)
@defNLExpr(bar[ink=1:2, indigo=1:2, igloo=1:2, icarus=1:2],
	       foo * ink * indigo * igloo * icarus)
[ ReverseDiffSparse.getvalue(bar[i, j, k, l], m.colVal) for i=1:2, j=1:2, k=1:2, l=1:2 ]


################

immutable SimpleThingy
	foo::Float64
	bar::Float64
	SimpleThingy(offset::Float64) = begin
		foo = offset
		bar = offset * 2
		new(foo, bar)
	end
end

immutable MyOtherThingy
	foo::Array{Float64}
	MyOtherThingy(offset::Float64) = begin
		foo = Array(Float64, 5, 10)
		for i = 1:5
			for j = 1:10
				foo[i, j] = offset + i + j
			end
		end
		new(foo)
	end
end

immutable My1DArrayThingy
	foo::Array{Float64}
	My1DArrayThingy(offset::Float64) = begin
		foo = Array(Float64, 10)
		for i = 1:10
			foo[i] = offset + i
		end
		new(foo)
	end
end

thingy_array = Array(SimpleThingy, 10)
for i = 1:10
	thingy_array[i] = SimpleThingy(convert(Float64, i))
end


thingy_array_array = Array(My1DArrayThingy, 10)
for i = 1:10
	thingy_array_array[i] = My1DArrayThingy(convert(Float64, i))
end


other_thingy_array = Array(MyOtherThingy, 10)
for i = 1:10
	other_thingy_array[i] = MyOtherThingy(convert(Float64, i))
end


m = Model()
@defVar(m, bar)
setValue(bar, 2)

# These work:
@defNLExpr(thingy_bar[i=1:10], thingy_array[i].foo * bar)
@defNLExpr(thingy_sum, sum{thingy_array[i].foo * bar, i=1:10})
ReverseDiffSparse.getvalue(thingy_sum, m.colVal)

# These do not:
@defNLExpr(other_thingy_bar[i=1:10], other_thingy_array[i].foo[1, 1] * bar)
@defNLExpr(other_thingy_sum, sum{other_thingy_array[i].foo[1, 1] * bar, i=1:10})
@defNLExpr(array_thingy_bar[i=1:5], thingy_array_array[i].foo[1] * bar)

# For references, these expressions work:
other_thingy_array[1].foo[1, 1]
thingy_array_array[1].foo[1]

# The way to do this is probably:
thingy_flattened = [ other_thingy_array[i].foo[1, 1] for i=1:10 ]
@defNLExpr(other_thingy_bar[i=1:10], thingy_flattened[i] * bar)

###########
# A simple example for the issue report:
immutable MyObject
	foo::Array{Float64}
	MyObject(x::Float64) = begin
		foo = Array(Float64, 10)
		for i = 1:10
			foo[i] = x + i
		end
		new(foo)
	end
end

object_array = [ MyObject(convert(Float64, i)) for i = 1:5]

m = Model()
@defVar(m, bar)
setValue(bar, 2)

# This expression gives the error
# ERROR: i not defined
@defNLExpr(object_bar[i=1:5, j=1:10], object_array[i].foo[j] * bar)

# This works:
object_flattened = [ object_array[i].foo[j] for i=1:5, j=1:10 ]
@defNLExpr(object_bar[i=1:5, j=1:10], object_flattened[i, j] * bar)

# This also works:
immutable MySimpleObject
	foo::Float64
	MySimpleObject(x::Float64) = begin
		foo = x
		new(foo)
	end
end
simple_object_array = [ MySimpleObject(convert(Float64, i)) for i = 1:5]
@defNLExpr(simple_object_bar[i=1:5], simple_object_array[i] * bar)



##########################
# Matrix transpose. This is failing in a strange way.

m = Model()
a = [[1, 2] [3, 4]]

@defNLExpr(matrix1[row=1:2, col=1:2], a[row, col]);

# Works:
matrix1_list_transpose =
	[ ReverseDiffSparse.getvalue(matrix1[col, row], m.colVal)
	  for row=1:2, col=1:2 ]

# Works:
@defNLExpr(matrix1_transpose_v2[r1=1:2, r2=1:2],
	       matrix1[r2, r1]);
matrix1_transpose_v2_value =
	[ ReverseDiffSparse.getvalue(matrix1_transpose_v2[row, col], m.colVal)
	  for row=1:2, col=1:2 ]

# Does not work:
@defNLExpr(matrix1_transpose[row=1:2, col=1:2],
	       matrix1[col, row]);
matrix1_transpose_value =
	[ ReverseDiffSparse.getvalue(matrix1_transpose[row, col], m.colVal)
	  for row=1:2, col=1:2 ]


##########################
# Matrix transpose. This is failing in a strange way.

m = Model()
a = [[1, 2] [3, 4]]

@defNLExpr(matrix1[rowi=1:2, coli=1:2], a[rowi, coli]);

# Works:
matrix_a =
	[ ReverseDiffSparse.getvalue(matrix1[rowi, coli], m.colVal)
	  for rowi=1:2, coli=1:2 ]

# Works:
matrix_a_t =
	[ ReverseDiffSparse.getvalue(matrix1[coli, rowi], m.colVal)
	  for rowi=1:2, coli=1:2 ]

# Does not work:
@defNLExpr(matrix1_transpose[rowi=1:2, coli=1:2],
	       matrix1[coli, rowi]);
matrix1_transpose_value =
	[ ReverseDiffSparse.getvalue(matrix1_transpose[rowi, coli], m.colVal)
	  for rowi=1:2, coli=1:2 ]

# Works:
@defNLExpr(matrix1_transpose_v2[r1=1:2, r2=1:2],
	       matrix1[r2, r1]);
matrix1_transpose_v2_value =
	[ ReverseDiffSparse.getvalue(matrix1_transpose_v2[rowi, coli], m.colVal)
	  for rowi=1:2, coli=1:2 ]


###############################
# sums.  Seems to work, maybe it was an indexing problem?

m = Model()
@defVar(m, foo)

x = [ 0.12321312321321, 0.4354354353245, 0.657457645674657, 0.143643514351251 ]
@defNLExpr(a, (x[1] * x[2]) + (x[3] * x[4]))
@defNLExpr(b, x[1] * x[2] + x[3] * x[4])

ReverseDiffSparse.getvalue(a, m.colVal)
ReverseDiffSparse.getvalue(b, m.colVal)

###################
# Ragged arrays of expressions.

m = Model()
@defVar(m, foo)

# This doesn't work:
my_set = [1, 2, 3]
@defNLExpr(in_my_set[i=1:5], i in my_set)

# This works though:
n = 10000
selection_matrix = zeros(Int64, n, n);
for i = 1:n
	selection_matrix[i, i] = 1
end
#selection_matrix = ones(Int64, n, n);

@defNLExpr(in_my_set[i=1:n, j=1:n],
	       sum{exp(-foo); selection_matrix[i, j] == 1});

@defNLExpr(sum_in_my_set, sum{in_my_set[i, j], i=1:n, j=1:n});
setValue(foo, 0.3)
ReverseDiffSparse.getvalue(sum_in_my_set, m.colVal)
n * exp(-0.3)







