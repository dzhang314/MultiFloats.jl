module MFIR

################################################################ MFIR OPERATIONS


export MFIROperation, MFIR_ABS, MFIR_NEG,
    MFIR_ADD, MFIR_TWO_SUM, MFIR_FAST_TWO_SUM,
    MFIR_SUB, MFIR_TWO_DIFF, MFIR_FAST_TWO_DIFF,
    MFIR_MUL, MFIR_FMA, MFIR_TWO_PROD, MFIR_INV, MFIR_DIV, MFIR_SQRT,
    cost, arity, num_outputs


@enum MFIROperation::UInt16 begin
    MFIR_ABS
    MFIR_NEG
    MFIR_ADD
    MFIR_TWO_SUM
    MFIR_FAST_TWO_SUM
    MFIR_SUB
    MFIR_TWO_DIFF
    MFIR_FAST_TWO_DIFF
    MFIR_MUL
    MFIR_FMA
    MFIR_TWO_PROD
    MFIR_INV
    MFIR_DIV
    MFIR_SQRT
end


@inline function cost(op::MFIROperation)
    if (op == MFIR_ABS) | (op == MFIR_NEG)
        return 0
    elseif (op == MFIR_ADD) | (op == MFIR_SUB)
        return 1
    elseif (op == MFIR_TWO_SUM) | (op == MFIR_TWO_DIFF)
        return 6
    elseif (op == MFIR_FAST_TWO_SUM) | (op == MFIR_FAST_TWO_DIFF)
        return 3
    elseif (op == MFIR_MUL) | (op == MFIR_FMA)
        return 2
    elseif (op == MFIR_TWO_PROD)
        return 4
    else # (op == MFIR_INV) | (op == MFIR_DIV) | (op == MFIR_SQRT)
        return 16
    end
end


@inline function arity(op::MFIROperation)
    if ((op == MFIR_ABS) | (op == MFIR_NEG) |
        (op == MFIR_INV) | (op == MFIR_SQRT))
        return 1
    elseif (op == MFIR_FMA)
        return 3
    else
        return 2
    end
end


@inline function num_outputs(op::MFIROperation)
    if ((op == MFIR_TWO_SUM) | (op == MFIR_TWO_DIFF) |
        (op == MFIR_FAST_TWO_SUM) | (op == MFIR_FAST_TWO_DIFF) |
        (op == MFIR_TWO_PROD))
        return 2
    else
        return 1
    end
end


##################################################### INSTRUCTION DATA STRUCTURE


export MFIRInstruction, arity, num_outputs, normalize


struct MFIRInstruction
    op::MFIROperation
    args::NTuple{3,UInt16}
end


const NULL_ARG = zero(UInt16)

@inline MFIRInstruction(op::MFIROperation, i::Integer) =
    MFIRInstruction(op, (i % UInt16, NULL_ARG, NULL_ARG))

@inline MFIRInstruction(op::MFIROperation, i::Integer, j::Integer) =
    MFIRInstruction(op, (i % UInt16, j % UInt16, NULL_ARG))

@inline MFIRInstruction(op::MFIROperation, i::Integer, j::Integer, k::Integer) =
    MFIRInstruction(op, (i % UInt16, j % UInt16, k % UInt16))


@inline function Base.isvalid(instruction::MFIRInstruction, num_available::Int)
    n = arity(instruction.op)
    a = instruction.args
    lo = one(UInt16)
    hi = num_available % UInt16
    @inbounds if n == 1
        return (lo <= a[1] <= hi) & iszero(a[2]) & iszero(a[3])
    elseif n == 2
        return (lo <= a[1] <= hi) & (lo <= a[2] <= hi) & iszero(a[3])
    elseif n == 3
        return (lo <= a[1] <= hi) & (lo <= a[2] <= hi) & (lo <= a[3] <= hi)
    else
        return false
    end
end


@inline arity(instruction::MFIRInstruction) = arity(instruction.op)
@inline num_outputs(instruction::MFIRInstruction) = num_outputs(instruction.op)


@inline function normalize(instruction::MFIRInstruction)
    op = instruction.op
    a = instruction.args
    if ((op == MFIR_ADD) | (op == MFIR_TWO_SUM) |
        (op == MFIR_MUL) | (op == MFIR_FMA) | (op == MFIR_TWO_PROD))
        return MFIRInstruction(op, (minmax(a[1], a[2])..., a[3]))
    else
        return instruction
    end
end


######################################################### PROGRAM DATA STRUCTURE


export MFIRProgram, num_registers, definition_map, use_counts


struct MFIRProgram
    num_inputs::Int
    instructions::Vector{MFIRInstruction}
    result_ranges::Vector{UnitRange{UInt16}}
    output_indices::Vector{UInt16}
end


function MFIRProgram(
    num_inputs::Integer,
    instructions::Vector{MFIRInstruction},
    output_indices::AbstractVector{<:Integer},
)
    hi = UInt16(num_inputs) # range check
    result_ranges = Vector{UnitRange{UInt16}}(undef, length(instructions))
    for (i, instruction) in enumerate(instructions)
        lo = hi + one(UInt16)
        hi = UInt16(hi + num_outputs(instruction)) # range check
        @inbounds result_ranges[i] = lo:hi
    end
    for index in output_indices
        @assert 1 <= index <= hi
    end
    return MFIRProgram(Int(num_inputs), instructions, result_ranges,
        convert(Vector{UInt16}, output_indices))
end


@inline function Base.isvalid(p::MFIRProgram)
    if !(0 <= p.num_inputs <= typemax(UInt16))
        return false
    end
    if length(p.instructions) != length(p.result_ranges)
        return false
    end
    num_available = p.num_inputs
    for (instruction, range) in zip(p.instructions, p.result_ranges)
        if !isvalid(instruction, num_available)
            return false
        end
        if range.start != num_available + 1
            return false
        end
        num_available += num_outputs(instruction)
        if range.stop != num_available
            return false
        end
    end
    if !(num_available <= typemax(UInt16))
        return false
    end
    for index in p.output_indices
        if !(1 <= index <= num_available)
            return false
        end
    end
    return true
end


@inline num_registers(p::MFIRProgram) =
    isempty(p.result_ranges) ? p.num_inputs : Int(p.result_ranges[end].stop)


function definition_map(p::MFIRProgram)
    result = zeros(Int, num_registers(p))
    for (i, range) in enumerate(p.result_ranges)
        result[range] .= i
    end
    return result
end


function use_counts(p::MFIRProgram)
    result = zeros(Int, num_registers(p))
    for instr in p.instructions
        for j in 1:arity(instr)
            result[instr.args[j]] += 1
        end
    end
    for out in p.output_indices
        result[out] += 1
    end
    return result
end


############################################################### PROGRAM MUTATION


export change_outputs, append_instruction, replace_instruction


change_outputs(p::MFIRProgram, output_index::Integer) =
    MFIRProgram(p.num_inputs, p.instructions, p.result_ranges,
        [UInt16(output_index)])


change_outputs(p::MFIRProgram, output_indices::AbstractVector{<:Integer}) =
    MFIRProgram(p.num_inputs, p.instructions, p.result_ranges,
        convert(Vector{UInt16}, output_indices))


function append_instruction(p::MFIRProgram, instruction::MFIRInstruction)
    instructions = push!(copy(p.instructions), instruction)
    n = num_registers(p)
    lo = (n + 1) % UInt16
    hi = (n + num_outputs(instruction)) % UInt16
    result_ranges = push!(copy(p.result_ranges), lo:hi)
    return MFIRProgram(p.num_inputs, instructions, result_ranges,
        p.output_indices)
end


function replace_instruction(
    p::MFIRProgram,
    index::Int,
    instruction::MFIRInstruction,
)
    @assert arity(instruction) == arity(p.instructions[index])
    instructions = copy(p.instructions)
    instructions[index] = instruction
    return MFIRProgram(p.num_inputs, instructions, p.result_ranges,
        p.output_indices)
end


################################################################################

end # module MFIR
