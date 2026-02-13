"""
Data formatting functions for finetuning.
These functions are used to format the data so that unsloth gets a single 'text' field.
"""

FORMATTING_FUNCS = {}

def register(name: str):
    def decorator(func):
        FORMATTING_FUNCS[name] = func
        return func
    return decorator

@register("deepmind_code_contests")
def format_deepmind_code_contests(example: dict) -> str:
    """
    Format the deepmind code contests dataset so that unsloth gets a single 'text' field.
    """
    # print("type of solutions: ", type(example['solutions']))
    # print("type of solution: ", type(example['solutions']['solution']))
    solutions = example['solutions']
    if isinstance(solutions, list):
        return [f"Description of the problem: {example['description']}\nSolution: {i}" for i in solutions]
    elif isinstance(solutions, dict) and 'solution' in solutions:
        if isinstance(solutions['solution'], list):
            return [f"Description of the problem: {example['description']}\nSolution: {i}" for i in solutions['solution']]
        elif isinstance(solutions['solution'], str):
            return [f"Description of the problem: {example['description']}\nSolution: {solutions['solution']}"]
        else:
            raise ValueError(f"Invalid solution type: {type(solutions['solution'])}")
    else:
        raise ValueError(f"Invalid solutions type: {type(solutions)}")
