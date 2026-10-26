import ast
import json
import re
import sys


def check_tool_inputs(root, workload, spec, paths):
    sys.path.insert(0, str(root / "scripts"))
    from gpt2_tokenizer import decode_tokens, load_encoder

    tokens = [int(value) for value in paths["tokens"].read_text().split()]
    mask = "".join(line.split("#", 1)[0].strip()
                   for line in paths["mask"].read_text().splitlines())
    parts = re.fullmatch(r"(0+)(1+)(0+)(1+)", mask)
    if len(tokens) != spec["length"] or len(mask) != len(tokens) or parts is None:
        raise ValueError("Invalid bundled transcript length or generation mask")
    decoder = load_encoder(root / "backup")
    if any(token not in decoder for token in tokens):
        raise ValueError("Unknown token ID in bundled transcript")
    call_start, call_end = parts.span(2)
    observation_start, observation_end = parts.span(3)
    call_text = decode_tokens(tokens[call_start:call_end], decoder).strip()
    observation = decode_tokens(tokens[observation_start:observation_end], decoder).strip()
    call = json.loads(call_text)
    field = "city" if workload == "weather" else "code"
    action = "weather" if workload == "weather" else "code"
    if (not isinstance(call, dict) or set(call) != {"action", field}
            or call["action"] != action or not isinstance(call[field], str)):
        raise ValueError("Generated tool call does not match the workload schema")
    if paths["document"].read_text().strip() != call_text:
        raise ValueError("Coral document does not match the generated tool call")
    result = {"tool_call": call, "token_range": [call_start, call_end]}
    if workload == "weather":
        if call["city"] != spec["city"]:
            raise ValueError("Weather request city does not match the generated tool call")
    else:
        check_coding_program(call["code"], paths["program"].read_text())
        observed = re.fullmatch(r"OBSERVATION:\s*(-?\d+)", observation)
        if observed is None:
            raise ValueError("Coding observation must contain one integer result")
        result["expected_output"] = int(observed[1])
    return result


def check_coding_program(code, program):
    body = ast.parse(code).body
    call = body[0].value if len(body) == 1 and isinstance(body[0], ast.Expr) else None
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
            or call.func.id != "print" or len(call.args) != 1 or call.keywords):
        raise ValueError("Coding demo requires exactly print(integer + integer)")
    expr = call.args[0]
    if (not isinstance(expr, ast.BinOp) or not isinstance(expr.op, ast.Add)
            or not all(isinstance(value, ast.Constant) and type(value.value) is int
                       and 0 <= value.value <= 2147483647 for value in (expr.left, expr.right))
            or expr.left.value + expr.right.value > 2147483647):
        raise ValueError("Coding demo requires a non-overflowing i32 integer addition")
    expected = ast.parse("def add() -> int:\n    return " + ast.unparse(expr) + "\n")
    if ast.dump(ast.parse(program)) != ast.dump(expected):
        raise ValueError("SP1 program does not match the generated coding expression")
