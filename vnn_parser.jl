# Given a verification specification in a .vnnlib file, convert to input and output polytope specs
using LinearAlgebra

function parse_vnn(filename)
    strarr = readlines(filename)

    x_var = r"declare-const X"
    y_var = r"declare-const Y"

    # All input constraints are boxes
    le_single_x = r"assert \(<= X"
    ge_single_x = r"assert \(>= X"

    # Output constraints
    le_single_y = r"\(<= Y"
    ge_single_y = r"\(>= Y"

    begin_output_con = r"assert \( or"

    num_inputs = 0
    num_outputs = 0

    num_input_cons = 0
    num_output_cons = 0

    lbs = Vector{Float64}(undef, 0)
    ubs = Vector{Float64}(undef, 0)

    label = parse(Int64, strarr[1][end-1])

    for line in strarr
        if occursin(x_var, line)
            num_inputs += 1
        elseif occursin(y_var, line)
            num_outputs += 1
        elseif occursin(le_single_x, line)
            num_input_cons += 1
            x_num = Int(floor(0.5*(num_input_cons-1)))
            push!(ubs, parse(Float64, line[14 + ndigits(x_num) + 1 : end-2]))
        elseif occursin(ge_single_x, line)
            num_input_cons += 1
            x_num = Int(floor(0.5*(num_input_cons-1)))
            push!(lbs, parse(Float64, line[14 + ndigits(x_num) + 1 : end-2]))
        elseif occursin(le_single_y, line)
            num_output_cons += 1
        elseif occursin(ge_single_y, line)
            num_output_cons += 1
        end
    end

    @show label

    # Sanity check
    if num_inputs != 784 || num_outputs != 10 || num_input_cons != 1568 || num_output_cons != 9 || length(lbs) != 784 || length(ubs) != 784
        error("Incorrect mnistfc vnnlib parsing.")
    end
  
    # Turn input upper and lower bounds into Ax≤b
    Aᵢ = vcat(Matrix{Float64}(-I, 784, 784), Matrix{Float64}(I, 784, 784))
    bᵢ = vcat(-lbs, ubs)

    A_out = Vector{Matrix{Float64}}(undef, 9)
    b_out = Vector{Vector{Float64}}(undef, 9)

    # Find Ay≤b given label
    idx = 1
    for i in 1:10
        if i != label
            a = zeros(10)
            a[i] = -1.
            a[label] = 1.
            A_out[idx] = reshape(a, (1,:))
            b_out[idx] = [0.]
            idx += 1
        end
    end

    return [Aᵢ], [bᵢ], A_out, b_out
end