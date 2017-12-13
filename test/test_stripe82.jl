import JLD

import Celeste.ParallelRun: BoundingBox,
    one_node_joint_infer, one_node_single_infer, detect_sources
import Celeste.SensitiveFloats: SensitiveFloat
import Celeste.DeterministicVI.ElboMaximize
import Celeste.Coordinates: match_coordinates

import FITSIO

"""
load_ea_from_source
Helper function to load elbo args for a particular source
"""
function load_ea_from_source(target_source, target_sources, catalog, images, all_vps)

    patches = Model.get_sky_patches(images, catalog)

    # Get neighbors of the source from which to load elbo args.
    neighbor_ids = Model.find_neighbors(patches, target_source)

    # Create a dictionary to the optimized parameters.
    target_source_variational_params = Dict{Int64, Array{Float64}}()
    for (indx, cur_source) in enumerate(target_sources)
        target_source_variational_params[cur_source] = all_vps[indx]
    end

    patches = Model.get_sky_patches(images, catalog)
    
    # Get neighbors of the source from which to load elbo args.
    neighbor_ids = Model.find_neighbors(patches, target_source)

    # Load neighbors, patches and variational parameters
    entry = catalog[target_source]
    neighbors = catalog[neighbor_ids]
    ids_local = vcat([target_source], neighbor_ids)
    cat_local = vcat([entry], neighbors)
    patches = patches[ids_local, :]  # limit patches to catalog
    vp = [haskey(target_source_variational_params, x) ?
          target_source_variational_params[x] :
          DeterministicVI.catalog_init_source(catalog[x]) for x in ids_local]

    # Create elbo args
    ElboArgs(images, patches, [1])
end

"""
compute_unconstrained_gradient
"""
function compute_unconstrained_gradient(target_source, target_sources, catalog, images, all_vps)
    # Load ea
    ea = load_ea_from_source(target_source, target_sources, catalog, images,
                                    all_vps)

    # Evaluate in constrained space and then unconstrain (taken from the old maximize_elbo.jl code)
    last_sf::SensitiveFloat{Float64} = SensitiveFloats.SensitiveFloat{Float64}(length(UnconstrainedParams), 1, true, true)
    transform = DeterministicVI.get_mp_transform(vp, ea.active_sources)
    f_res = DeterministicVI.elbo(ea)
    Transform.transform_sensitive_float!(transform, last_sf, f_res, vp, ea.active_sources)
    last_sf.d
end

"""
unconstrained_gradient_near_zero
"""
function unconstrained_gradient_near_zero(target_sources, catalog, images, all_vps)

    # Compute gradient per source
    for (cur_source_indx, source) in enumerate(target_sources)
        gradient = compute_unconstrained_gradient(source, target_sources, catalog, images,all_vps)
        if !isapprox(gradient, zeros(length(gradient)), atol=1)
            return false
        end
    end
    return true
end

"""
compare_vp_params
Helper to check whether two sets of variational parameters from the
result of joint inference are the same.

Return true if vp params are the same, false otherwise
"""
function compare_vp_params(flux_loc, flux_scale)

    length(flux_loc) == length(flux_scale) || return false

    # Check the existence and equivalence of each source's vp in flux_scale
    for i in eachindex(flux_loc)
        a = flux_loc[i].vs
        b = flux_scale[i].vs
        if !(isapprox(a, b))
            println("compare_vp_params: Mismatch - $(a) vs $(b)")
            print("norm(a - b): ", norm(a - b))
            return false
        end
    end
    return true
end

"""
compute_obj_value
computes obj value given set of results from one_node_infer and one_node_joint_infer
"""
function compute_obj_value(results,
                           rcfs::Vector{RunCamcolField},
                           stagedir::String;
                           box=BoundingBox(-1000.,1000.,-1000.,1000.),
                           primary_initialization=true)
    catalog, target_sources = infer_init(rcfs, stagedir; box=box, primary_initialization=primary_initialization)
    images = SDSSIO.load_field_images(rcfs, stagedir)

    vp = [r.vs for r in results]

    # it may be better to pass `patches` as an argument to `compute_obj_value`.
    patches = Model.get_sky_patches(images, catalog[target_sources])

    # this works since we're just generating patches for active_sources.
    # if instead you pass patches for all sources, then instead we'd used
    # active_sources = target_sources
    active_sources = collect(1:length(target_sources))

    ea = ElboArgs(images, patches, active_sources,
                  calculate_gradient=false, calculate_hessian=false)
    DeterministicVI.elbo(ea).v[]
end

"""
load_stripe_82_data
"""
function load_stripe_82_data()
    rcf = RunCamcolField(4263, 5, 119)
    cd(datadir)
    run(`make RUN=$(rcf.run) CAMCOL=$(rcf.camcol) FIELD=$(rcf.field)`)
    cd(wd)
    catalog, target_sources = infer_init([rcf], datadir, primary_initialization=false)
    images = SDSSIO.load_field_images([rcf], datadir)
    ([rcf], datadir, target_sources, catalog, images)
end


"""
test_improve_stripe_82_obj_value
"""
function test_improve_stripe_82_obj_value()
    println("Testing that joint_infer improves score on stripe 82...")
    (rcfs, datadir, target_sources, catalog, images) = load_stripe_82_data()

    ctni = ParallelRun.infer_init(rcfs, strategy; box=box)

    # Single inference obj value
    infer_single(ctni...) = one_node_single_infer(ctni...)
    result_single = one_node_infer(rcfs, datadir;
                                   primary_initialization=false)
    score_single = compute_obj_value(result_single, rcfs, datadir;
                                     primary_initialization=false)

    # Joint inference obj value
    infer_multi(ctni...) = one_node_joint_infer(ctni...;
                                                n_iters=30,
                                                within_batch_shuffling=true)
    result_multi = one_node_infer(rcfs, datadir;
                                  infer_callback=infer_multi,
                                  primary_initialization=false)
    score_multi = compute_obj_value(result_multi, rcfs, datadir;
                                    primary_initialization=false)

    println("Score single: $(score_single)")
    println("Score multi: $(score_multi)")

    @test score_multi > score_single
end

"""
test_gradient_is_near_zero_on_stripe_82
TODO(max) - This doesn't pass right now since some light sources are not close to zero.
"""
function test_gradient_is_near_zero_on_stripe_82()
    (rcfs, datadir, target_sources, catalog, images) = load_stripe_82_data()

    # Make sure joint infer with few iterations does not pass the gradient near zero check
    joint_few(cnti...) = one_node_joint_infer(cnti...)
    results_few = one_node_infer(rcfs, datadir; infer_callback=joint_few, primary_initialization=false)
    @test !unconstrained_gradient_near_zero(target_sources, catalog, images, [x.vs for x in results_few])

    # Make sure joint infer with many iterations passes the gradient near zero check
    joint_many(cnti...) = one_node_joint_infer(cnti...; n_iters=10)
    results_many = one_node_infer(rcfs, datadir; infer_callback=joint_many, primary_initialization=false)
    @test unconstrained_gradient_near_zero(target_sources, catalog, images, [x.vs for x in results_many])
end

"""
test_gradient_is_near_zero_on_four_sources
Tests that the gradient is zero on four sources after joint optimization.
Makes sure that with fewer iterations the gradient is not zero.
"""
function test_gradient_is_near_zero_on_four_sources()
    box = BoundingBox(154.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    catalog, target_sources = infer_init(field_triplets, datadir; box=box)
    images = SDSSIO.load_field_images(field_triplets, datadir)

    infer_few(ctni...) = one_node_joint_infer(ctni...;
                                              n_iters=2,
                                              within_batch_shuffling=true)
    result_few = one_node_infer(field_triplets, datadir;
                                infer_callback=infer_few,
                                box=box)
    @test !unconstrained_gradient_near_zero(target_sources, catalog, images, [x.vs for x in result_few])

    infer_many(ctni...) = one_node_joint_infer(ctni...;
                                               n_iters=100,
                                               within_batch_shuffling=true)
    result_many = one_node_infer(field_triplets, datadir;
                                 infer_callback=infer_many,
                                 box=box)
    @test unconstrained_gradient_near_zero(target_sources, catalog, images, [x.vs for x in result_many])

end

"""
test_different_result_with_different_iter()
Using 3 iters instead of 1 iters should result in a different set of parameters.
"""
function test_different_result_with_different_iter()
    # This bounding box has overlapping stars. (neighbor map is not empty)
    box = BoundingBox(154.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    infer_few(ctni...) = one_node_joint_infer(ctni...;
                                              n_iters=1,
                                              within_batch_shuffling=false)
    result_iter_1 = one_node_infer(field_triplets, datadir;
                                   infer_callback=infer_few, box=box)

    infer_many(ctni...) = one_node_joint_infer(ctni...;
                                               n_iters=5,
                                               within_batch_shuffling=false)
    result_iter_5 = one_node_infer(field_triplets, datadir;
                                   infer_callback=infer_many, box=box)

    # Make sure that parameters are not the same
    @test !compare_vp_params(result_iter_1, result_iter_5)
end

"""
test_same_result_with_diff_batch_sizes
Varying batch sizes using the cyclades algorithm should not change final objective value.
"""
function test_same_result_with_diff_batch_sizes()
    # This bounding box has overlapping stars. (neighbor map is not empty)
    box = BoundingBox(154.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    # With batch size = 7
    infer_few(ctni...) = one_node_joint_infer(ctni...;
                                n_iters=3,
                                batch_size=7,
                                within_batch_shuffling=false)
    result_bs_7 = one_node_infer(field_triplets, datadir;
                                 infer_callback=infer_few,
                                 box=box)


    # With batch size = 39
    infer_many(ctni...) = one_node_joint_infer(ctni...;
                                n_iters=3,
                                batch_size=39,
                                within_batch_shuffling=false)
    result_bs_39 = one_node_infer(field_triplets, datadir;
                                  infer_callback=infer_many,
                                  box=box)

    # Make sure that parameters are exactly the same
    @test compare_vp_params(result_bs_7, result_bs_39)
end

function test_same_one_node_infer_twice()
    # This bounding box has overlapping stars. (neighbor map is not empty)
    box = BoundingBox(154.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    infer_few(ctni...) = one_node_joint_infer(ctni...;
                                              n_iters=3,
                                              batch_size=7,
                                              within_batch_shuffling=false)
    result_bs_7_1 = one_node_infer(field_triplets, datadir;
                                   infer_callback=infer_few,
                                   box=box)

    result_bs_7_2 = one_node_infer(field_triplets, datadir;
                                   infer_callback=infer_few,
                                   box=box)

    # Make sure that parameters are exactly the same
    @test compare_vp_params(result_bs_7_1, result_bs_7_2)
end

"""
test infer multi iter with a single (run, camcol, field).
This is basically just to make sure it runs at all.
"""
function test_one_node_joint_infer()
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    box = BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]
    result = one_node_infer(field_triplets, datadir;
                            infer_callback=one_node_joint_infer,
                            box=box)
end


"""
test infer multi iter obj overlapping.
This makes sure one_node_joint_infer achieves sum objective value less
than single iter on non overlapping sources.
"""
function test_one_node_joint_infer_obj_overlapping()
    # This bounding box has overlapping stars. (neighbor map is not empty)
    box = BoundingBox(154.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    # 100 iterations
    tic()
    infer_multi(ctni...) = one_node_joint_infer(ctni...;
                                                n_iters=30,
                                                within_batch_shuffling=true)
    result_multi = one_node_infer(field_triplets, datadir;
                                  infer_callback=infer_multi,
                                  box=box)
    multi_iter_time = toq()
    score_multi = compute_obj_value(result_multi, field_triplets, datadir; box=box)

    # 2 iterations
    tic()
    infer_two(ctni...) = one_node_joint_infer(ctni...;
                                              n_iters=2,
                                              within_batch_shuffling=true)
    result_two = one_node_infer(field_triplets, datadir;
                                infer_callback=infer_two,
                                box=box)
    two_iter_time = toq()
    score_two = compute_obj_value(result_two, field_triplets, datadir; box=box)

    # One node infer (1 iteration, butm ore newton steps)
    tic()
    result_single = one_node_infer(field_triplets, datadir;
                                   infer_callback=one_node_single_infer,
                                   box=box)
    single_iter_time = toq()
    score_single = compute_obj_value(result_single, field_triplets, datadir; box=box)

    println("One node joint infer objective value: $(score_multi)")
    println("One node joint infer 2 iter objective value: $(score_two)")
    println("Single iter objective value: $(score_single)")
    println("One node joint infer time: $(multi_iter_time)")
    println("One node joint infer 2 iter time: $(two_iter_time)")
    println("Single iter time: $(single_iter_time)")
    @test score_multi > score_single
    @test score_multi > score_two
end


# Stripe 82 tests are long running
@testset "stripe82"
    test_improve_stripe_82_obj_value()

    # Test gradients near zero (this takes 100 iterations and is a bit slow)
    test_gradient_is_near_zero_on_four_sources()

    # Test that we reach a higher objective with more iterations.
    # This is a bit slow.
    test_one_node_joint_infer_obj_overlapping()

    test_same_one_node_infer_twice()
    test_different_result_with_different_iter()
    test_same_result_with_diff_batch_sizes()
    test_one_node_joint_infer()
end