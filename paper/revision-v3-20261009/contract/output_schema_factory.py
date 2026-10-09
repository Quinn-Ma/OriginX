def output_schema(ids, tokens):
    return {"type": "object", "properties": {"requests": {"type": "array",
            "minItems": len(ids), "maxItems": len(ids), "items": {"type": "object",
            "properties": {"request_id": {"type": "string", "enum": ids},
                           "request_token": {"type": "string", "enum": list(tokens.values())},
                           "subgoal_instruction": {"type": "string", "minLength": 1, "maxLength": 1500},
                           "rationale": {"type": "string", "maxLength": 1500}},
            "required": ["request_id", "request_token", "subgoal_instruction", "rationale"],
            "additionalProperties": False}}}, "required": ["requests"], "additionalProperties": False}
