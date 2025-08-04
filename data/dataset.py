def apply_alpaca_template(instruction, input):
    if input != None:
        template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:{instruction}\n### Input:{input}\n### Response:'''
    else:
        template = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Response:'''
    prompt = template.format(instruction=instruction, input=input)
    return prompt