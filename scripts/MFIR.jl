module MFIR

################################################################ MFIR OPERATIONS


export MFIROperation, MFIR_ABS, MFIR_NEG,
    MFIR_ADD, MFIR_TWO_SUM, MFIR_FAST_TWO_SUM,
    MFIR_SUB, MFIR_TWO_DIFF, MFIR_FAST_TWO_DIFF,
    MFIR_SQR, MFIR_MUL, MFIR_FMA, MFIR_TWO_SQR, MFIR_TWO_PROD,
    MFIR_INV, MFIR_DIV, MFIR_SQRT,
    arity, num_outputs


@enum MFIROperation::UInt16 begin
    MFIR_ABS
    MFIR_NEG
    MFIR_ADD
    MFIR_TWO_SUM
    MFIR_FAST_TWO_SUM
    MFIR_SUB
    MFIR_TWO_DIFF
    MFIR_FAST_TWO_DIFF
    MFIR_SQR
    MFIR_MUL
    MFIR_FMA
    MFIR_TWO_SQR
    MFIR_TWO_PROD
    MFIR_INV
    MFIR_DIV
    MFIR_SQRT
end


@inline function arity(op::MFIROperation)
    if ((op == MFIR_ABS) | (op == MFIR_NEG) |
        (op == MFIR_SQR) | (op == MFIR_TWO_SQR) |
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
        (op == MFIR_TWO_SQR) | (op == MFIR_TWO_PROD))
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
    if signbit(num_available)
        return false
    end
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
        @assert isvalid(instruction, Int(hi))
        lo = hi + one(UInt16)
        hi = UInt16(hi + num_outputs(instruction)) # range check
        @inbounds result_ranges[i] = lo:hi
    end
    for index in output_indices
        @assert 1 <= index <= hi
    end
    return MFIRProgram(
        Int(num_inputs),
        instructions,
        result_ranges,
        convert(Vector{UInt16}, output_indices))
end


@inline function Base.isvalid(program::MFIRProgram)
    if !(0 <= program.num_inputs <= typemax(UInt16))
        return false
    end
    if length(program.instructions) != length(program.result_ranges)
        return false
    end
    num_available = program.num_inputs
    for (instruction, range) in zip(program.instructions, program.result_ranges)
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
    for index in program.output_indices
        if !(1 <= index <= num_available)
            return false
        end
    end
    return true
end


@inline num_registers(program::MFIRProgram) =
    isempty(program.result_ranges) ? program.num_inputs :
    Int(program.result_ranges[end].stop)


@inline num_registers(program::MFIRProgram, index::Integer) =
    iszero(index) ? program.num_inputs :
    Int(program.result_ranges[index].stop)


function definition_map(program::MFIRProgram)
    result = zeros(Int, num_registers(program))
    for (i, range) in enumerate(program.result_ranges)
        result[range] .= i
    end
    return result
end


function use_counts(program::MFIRProgram)
    result = zeros(Int, num_registers(program))
    @inbounds for instr in program.instructions
        for j in 1:arity(instr)
            result[instr.args[j]] += 1
        end
    end
    @inbounds for out in program.output_indices
        result[out] += 1
    end
    return result
end


############################################################### PROGRAM MUTATION


export change_outputs, append_instruction, replace_instruction


function change_outputs(program::MFIRProgram, output_index::Integer)
    @assert 1 <= output_index <= num_registers(program)
    return MFIRProgram(
        program.num_inputs,
        program.instructions,
        program.result_ranges,
        [output_index % UInt16])
end


function change_outputs(
    program::MFIRProgram,
    output_indices::AbstractVector{<:Integer},
)
    n = num_registers(program)
    for output_index in output_indices
        @assert 1 <= output_index <= n
    end
    return MFIRProgram(
        program.num_inputs,
        program.instructions,
        program.result_ranges,
        convert(Vector{UInt16}, output_indices))
end


function append_instruction(program::MFIRProgram, instruction::MFIRInstruction)
    n = num_registers(program)
    @assert isvalid(instruction, n)
    instructions = push!(copy(program.instructions), instruction)
    lo = (n + 1) % UInt16
    hi = (n + num_outputs(instruction)) % UInt16
    result_ranges = push!(copy(program.result_ranges), lo:hi)
    return MFIRProgram(
        program.num_inputs,
        instructions,
        result_ranges,
        program.output_indices)
end


function replace_instruction(
    program::MFIRProgram,
    index::Int,
    instruction::MFIRInstruction,
)
    @assert 1 <= index <= length(program.instructions)
    @inbounds old_instruction = program.instructions[index]
    @assert num_outputs(instruction) == num_outputs(old_instruction)
    @assert isvalid(instruction, num_registers(program, index - 1))
    instructions = copy(program.instructions)
    @inbounds instructions[index] = instruction
    return MFIRProgram(
        program.num_inputs,
        instructions,
        program.result_ranges,
        program.output_indices)
end


function remove_instruction(
    program::MFIRProgram,
    index::Int,
    replacements::NTuple{N,UInt16};
    outputs::AbstractVector{UInt16}=program.output_indices,
) where {N}
    old_num_instructions = length(program.instructions)
    @assert 1 <= index <= old_num_instructions
    @inbounds removed = program.instructions[index]
    @assert num_outputs(removed) == N
    @inbounds removed_base = first(program.result_ranges[index])
    for replacement in replacements
        @assert 1 <= replacement < removed_base
    end
    old_num_registers = num_registers(program)
    for output in outputs
        @assert 1 <= output <= old_num_registers
    end

    remap = Vector{UInt16}(undef, old_num_registers)
    @inbounds for i = one(UInt16):removed_base-one(UInt16)
        remap[i] = i
    end
    @inbounds for j = 1:N
        remap[removed_base+j-1] = replacements[j]
    end

    instructions = Vector{MFIRInstruction}(undef, old_num_instructions - 1)
    result_ranges = Vector{UnitRange{UInt16}}(undef, old_num_instructions - 1)
    base_index = program.num_inputs
    @inbounds for i = 1:old_num_instructions
        if i != index
            old_instruction = program.instructions[i]
            op = old_instruction.op
            new_args = map(
                arg -> (arg == NULL_ARG) ? NULL_ARG : @inbounds(remap[arg]),
                old_instruction.args)
            new_instruction = MFIRInstruction(op, new_args)
            new_index = i - (i > index)
            instructions[new_index] = new_instruction
            next_base_index = base_index + num_outputs(op)
            lo = (base_index + 1) % UInt16
            hi = next_base_index % UInt16
            result_ranges[new_index] = lo:hi
            for (j, out) in enumerate(program.result_ranges[i])
                remap[out] = (base_index + j) % UInt16
            end
            base_index = next_base_index
        end
    end

    new_outputs = Vector{UInt16}(undef, length(outputs))
    @inbounds for (i, output) in enumerate(outputs)
        new_outputs[i] = remap[output]
    end
    return MFIRProgram(
        program.num_inputs,
        instructions,
        result_ranges,
        new_outputs)
end


################################################################################

end # module MFIR
