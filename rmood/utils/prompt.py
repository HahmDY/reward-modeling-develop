LABELING_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by four AI assistants to the user question displayed below.
You should rank the assistants based on how well each follows the user's instructions and answers the user's question.
Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, level of detail, and harmlessness of their responses.
Begin your evaluation by briefly comparing two responses (A, B). Provide a concise explanation highlighting the strengths and weaknesses of each response.
Avoid any positional biases and ensure that the order in which the responses were presented does not influence your decision. 
Be as objective and impartial as possible.

[Format]
Output your final ranking by strictly following this format. Do not modify the tags, and be sure to generate the answer between the tags:
<thought>
Brief explanation of your ranking
</thought>
<ranking_1>
Letter of the best assistant
</ranking_1>
<ranking_2>
Letter of the second best assistant
</ranking_2>

[Conversation between User and Assistant]
{conversation}

[Response from Assistant A]
{assistant_a}

[Response from Assistant B]
{assistant_b}
"""
