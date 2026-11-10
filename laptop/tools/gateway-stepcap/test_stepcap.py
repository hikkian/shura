import re, types
src = open('ai_gateway.py.new').read()
i = src.index('BUDGET_MESSAGE = ('); j = src.index('def local_title')
ns = {'M': {'models': {'m': {'effortBudgets': {'low': 2048, 'xhigh': 32768, 'max': -1}, 'effortStepCaps': {'xhigh': 12288, 'max': 24576}},
                       'plain': {'effortBudgets': {'xhigh': 32768}}}}}
exec(src[i:j], ns)
ab = ns['apply_effort_budget']
U = {'role': 'user', 'content': 'x'}; A = {'role': 'assistant', 'content': '', 'tool_calls': [{}]}; T = {'role': 'tool', 'content': 'ok'}
def run(model, effort, msgs, **kw):
    o = {'reasoning_effort': effort, 'messages': msgs, **kw}; ab(o, model); return o.get('reasoning_budget_tokens')
assert run('m', 'xhigh', [U]) == 32768                 # first step keeps the full budget
assert run('m', 'xhigh', [U, A, T]) == 12288           # later step capped
assert run('m', 'xhigh', [U, A, T, A, T]) == 12288
assert run('m', 'max', [U]) == -1                      # max unlimited on the first step
assert run('m', 'max', [U, A, T]) == 24576             # but capped later
assert run('m', 'low', [U, A, T]) == 2048              # no cap configured for low, unchanged
assert run('m', 'xhigh', [U, A, T, U]) == 32768        # new user message = new first step
assert run('plain', 'xhigh', [U, A, T]) == 32768       # model without effortStepCaps: unchanged
assert run('m', 'xhigh', [U, A, T], reasoning_budget_tokens=7) == 7  # client value wins
print('all ok')
