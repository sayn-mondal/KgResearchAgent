def to_dict(value):
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)

    return {}


def make_usage_record(provider, model, step, usage):

    data = to_dict(usage)

    if provider == "groq":
        input_tokens = data.get("prompt_tokens")
        output_tokens = data.get("completion_tokens")
    else:
        input_tokens = data.get("input_tokens")
        output_tokens = data.get("output_tokens")

    reported = (
        input_tokens is not None
        and output_tokens is not None
    )

    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)

    total_tokens = int(
        data.get(
            "total_tokens",
            input_tokens + output_tokens
        ) or 0
    )

    return {
        "step": step,
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_reported": reported
    }


def append_usage(existing_usage, new_record):

    calls = list(existing_usage.get("calls", []))
    calls.append(new_record)

    result = {"calls": calls}

    for provider in ("groq", "gemini"):

        provider_calls = [
            call for call in calls
            if call["provider"] == provider
        ]

        result[provider] = {
            "input_tokens": sum(
                c["input_tokens"] for c in provider_calls
            ),
            "output_tokens": sum(
                c["output_tokens"] for c in provider_calls
            ),
            "total_tokens": sum(
                c["total_tokens"] for c in provider_calls
            ),
            "calls": len(provider_calls),
            "unreported_calls": sum(
                not c["usage_reported"]
                for c in provider_calls
            )
        }

    result["total"] = {
        "input_tokens": sum(
            result[p]["input_tokens"]
            for p in ("groq", "gemini")
        ),
        "output_tokens": sum(
            result[p]["output_tokens"]
            for p in ("groq", "gemini")
        ),
        "total_tokens": sum(
            result[p]["total_tokens"]
            for p in ("groq", "gemini")
        )
    }

    return result