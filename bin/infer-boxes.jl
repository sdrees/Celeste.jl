#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, infer_boxes
import Celeste.Log

function run_infer_boxes(args::Vector{String})
    if length(args) != 2
        println("""
            Usage:
              infer-boxes.jl <boxes_file> <out_dir>

            <boxes_file> format, one line per box:
            <difficulty>	<#RCFs>	<#sources>	<ramin> <ramax> <decmin> <decmax>
            """)
        exit(-1)
    end
    if !haskey(ENV, "CELESTE_STAGE_DIR")
        println("Set CELESTE_STAGE_DIR!")
        exit(-2)
    end

    # parse the boxes file
    all_boxes = BoundingBox[]
    box_source_counts = Int64[]
    boxes_file = args[1]
    f = open(boxes_file)
    for ln in eachline(f)
        lp = split(ln, '\t')
        if length(lp) != 4
            Log.one_message("ERROR: malformed line in box file, skipping ",
                            "remainder\n> $ln")
            break
        end
        sc = parse(Int64, lp[3])
        push!(box_source_counts, sc)
        ss = split(lp[4], ' ')
        ramin = parse(Float64, ss[1])
        ramax = parse(Float64, ss[2])
        decmin = parse(Float64, ss[3])
        decmax = parse(Float64, ss[4])
        bb = BoundingBox(ramin, ramax, decmin, decmax)
        push!(all_boxes, bb)
    end
    close(f)
    if length(all_boxes) < 1
        println("box file is empty?")
        exit(-1)
    end

    # run Celeste
    infer_boxes(all_boxes, box_source_counts, ENV["CELESTE_STAGE_DIR"], args[2])
end

run_infer_boxes(ARGS)

