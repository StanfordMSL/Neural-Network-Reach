# Given a verification specification in a .vnnlib file, convert to input and output polytope specs

function parse()
    strarr = readlines("prop_1.vnnlib")

    x_var = r"declare-const X"
    y_var = r"declare-const Y"

    # I think all input constraints are boxes
    le_single_x = r"assert \(<= X"
    ge_single_x = r"assert \(>= X"

    # Output constraints
    le_single_y = r"assert \(<= Y"
    ge_single_y = r"assert \(>= Y"

    and_con = r"assert \( and"
    or_con = r"assert \( or"

    num_inputs = 0
    num_outputs = 0

    num_input_cons = 0
    num_output_cons = 0

    for line in strarr
        if occursin(x_var, line)
            num_inputs += 1
        elseif occursin(y_var, line)
            num_outputs += 1
        elseif occursin(le_single_x, line)
            num_input_cons += 1
        elseif occursin(ge_single_x, line)
            num_input_cons += 1
        elseif occursin(le_single_y, line)
            num_output_cons += 1
        elseif occursin(ge_single_y, line)
            num_output_cons += 1
        elseif occursin(and_con, line)
            num_output_cons += 1
        elseif occursin(or_con, line)
            num_output_cons += 1
        end
    end

    @show num_inputs
    @show num_outputs
    @show num_input_cons
    @show num_output_cons
end

