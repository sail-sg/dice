FEWSHOT_TEMPLATE = """Given the fields `question`, produce the fields `answer`.

---

Follow the following format.

### Question: $[question]
### Answer: Let's think step by step in order to $[produce the answer]. We ...
Answer: $[answer]

---

### Question: Jackie spends 8 hours working, 3 hours of exercise and spends 8 hours of sleep per day. How much free time does Jackie have?
### Answer: Let's think step by step in order to calculate the free time Jackie has. We know that there are 24 hours in a day, and Jackie spends 8 hours working, 3 hours exercising, and 8 hours sleeping. Therefore, the total time spent is 8 + 3 + 8 = 19 hours. Since there are 24 hours in a day, the free time Jackie has is 24 - 19 = 5 hours.
Answer: 5

---

### Question: Jeremy buys 3 bags of chips for a party. Stacy and Emily also buy chips for the party. If they need 10 bags of chips total, and Stacy buys 4 bags, how many bags of chips should Emily buy?
### Answer: Let's think step by step in order to find the number of bags of chips Emily should buy. We know that Jeremy bought 3 bags, and Stacy bought 4 bags. So far, we have 7 bags. We need 10 bags total, so we need 3 more bags. Therefore, Emily should buy 3 bags.
Answer: 3

---

### Question: Heloise has dogs and cats in the ratio of 10:17, with the total number of pets being 189. If she gives 10 dogs to her friend Janet, how many dogs does she remain with altogether?
### Answer: Let's think step by step in order to find the number of dogs Heloise has. We know that the ratio of dogs to cats is 10:17, and the total number of pets is 189. This means that for every 10 dogs, there are 17 cats. If we divide 189 by 27 (10+17), we get 7. This means that for every 7 sets of 10 dogs and 17 cats, there are 189 pets. Since we want to find the number of dogs, we can multiply 7 by 10 to get 70. This means that for every 70 pets, there are 10 dogs. Since Heloise gave 10 dogs to Janet, she now has 70 less pets, which means she has 10 less dogs. Therefore, she has 70-10 = 60 dogs.
Answer: 60

---

### Question: Ben works 8-hour shifts in a furniture shop. It takes him 5 hours to build 1 rocking chair. How many chairs can he build in 10 days?
### Answer: Let's think step by step in order to find the number of chairs Ben can build in 10 days. We first find the number of hours Ben works in 10 days, which is 10 days * 8 hours/day = 80 hours. We then divide the total number of hours by the number of hours it takes to build 1 chair, which is 80 hours / 5 hours/chair = 16 chairs.
Answer: 16

---

### Question: {instruction}\n ### Answer: """

JUDGE_TEMPLATE = """Review the user's question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.

User: {instruction}
Response: {response}

After examining the user's instruction and the response, provide brief step-by-step justifications and conclude with a score between 0 to 5. You must follow the format below:

Evaluation: <evaluation>
Score: <score>"""

FEWSHOT_JUDGE_TEMPLATE = """Review the user's question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.

Follow the following format.

User: <question>
Response: <answer>
Evaluation: <evaluation>
Score: <score>

---

User: What is a Lemon Tree?
Response: The lemon (Citrus limon) is a species of small evergreen trees in the flowering plant family Rutaceae, native to Asia, primarily Northeast India (Assam), Northern Myanmar or China.\n\nIt was introduced into Spain and North Africa sometime between the years 1000 and 1200 ce. It was further distributed through Europe by the Crusaders, who found it growing in Palestine. In 1494 the fruit was being cultivated in the Azores and shipped largely to England. The lemon was thought by 18th-century Swedish botanist Carolus Linnaeus to be a variety of citron (Citrus medica), though it is now known to be a separate hybrid species.\n\nThe lemon forms an evergreen spreading bush or small tree, 3–6 meters high if not pruned. Its young oval leaves have a decidedly reddish tint; later they turn green. In some varieties the young branches of the lemon are angular; some have sharp thorns at the axils of the leaves. The flowers have a sweet odour and are solitary or borne in small clusters in the axils of the leaves. Reddish-tinted in the bud, the petals are usually white above and reddish purple below. The fruit is oval with a broad, low, apical nipple and forms 8 to 10 segments. The outer rind, or peel, yellow when ripe and rather thick in some varieties, is prominently dotted with oil glands. The white spongy inner part of the peel, called the mesocarp or albedo, is nearly tasteless and is the chief source of commercial grades of pectin. The seeds are small, ovoid, and pointed; occasionally fruits are seedless. The pulp is decidedly acidic.
Evaluation: The response provides a comprehensive overview of the lemon tree, covering its scientific classification, historical introduction to various regions, physical description, and the characteristics of its fruit. It is relevant and contains detailed information related to the inquiry, addressing the basic elements of what a lemon tree is. The answer is clearly written from an AI Assistant's perspective, offering a well-organized and thorough explanation, demonstrating a high level of knowledge on the topic. However, it might include slightly more detail than necessary for a basic understanding, but this does not detract significantly from its overall quality.
Score: 5

---

User: Say you are a person who has a vague interest in music and has vast experience as a consumer of music. If this person were to also be interested in learning about the process of music creation, what would be the best step-by-step plan for achieving this?
Response: Music is a good thing for humans. It can make you feel relaxed when you are tired.
Evaluation: The response lacks relevance to the user's query, failing to provide any step-by-step plan or guidance for learning about the process of music creation. While it acknowledges the positive effects of music, it doesn't address the user's specific inquiry or offer any actionable advice. The response is too vague and doesn't align with the user's request for a structured plan.
Score: 1

---

User: I have completed a Computer Science degree at a university and worked as a Data Scientists for two years at a start-up. I can code in C, C++, and Python. The latter is the one I use at work. I want to work at a big fund like Square Point as a quant researcher. What additional skills do I need to gain in order to qualify for such position?
Response: To work at a big fund like Square Point as a quant researcher, having experience in coding, especially in Python, is beneficial. This aligns with your current skill set from your computer science background and work as a data scientist. However, it's important to note that quant researchers often require a strong foundation in mathematics and statistics, which may not have been fully covered in your current experience.
Evaluation: The response is relevant to the user's inquiry, providing initial guidance on transitioning to a quant researcher role, thus meeting the first criterion. It addresses a substantial part of the user's question by highlighting the importance of a strong foundation in mathematics and statistics, satisfying the second criterion. The answer covers the basic elements of the question by connecting the user's current skills with required ones, earning the third point. However, it lacks depth, specific examples of additional skills needed (like machine learning, financial engineering, or specific mathematical concepts), and guidance on how to acquire these skills. This omission prevents the response from being comprehensive and directly tailored to the AI Assistant's perspective with actionable advice. It doesn't achieve the fourth and fifth criteria fully because it's not as detailed, organized, or insightful as it could be.
Score: 3

---

User: Write me a lua script for my roblox experience that will allow the character to run faster than the default.
Response: Sure, I can do that. Here is a simple Lua script that makes the player run faster than the default. Make sure to put it in a server script for it to work properly.\n```lua\nlocal player = "player_to_accelerate"\n\n-- default walk speed is 16\nlocal walk_speed = 24\n\ngame.Players[player].Character.Humanoid.WalkSpeed = walk_speed\n```\nThis takes a player, and sets their WalkSpeed to the value specified in the walk_speed variable. Make sure to replace the player variable with the player you want to speed up.
Evaluation: This response is relevant and provides a Lua script to address the user's request, increasing the character's speed in Roblox. It also explains how to implement the script. While it could have elaborated more on potential customization or additional features, it adequately addresses the user's inquiry with a clear and concise solution. The response demonstrates a good understanding of the user's needs and offers a practical solution. However, it could have provided more context or explanations for beginners.
Score: 4

---
User: What measurements can the developer of Open Assistant take to make their system more:\n\n1. Helpful\n2. Ethical\n3. Truthful\n\nDo you see scenarios where these 3 properties contradict each other?
Response: Not that much I think. There's already the option to flag and label almost everything. Sometimes people don't like the truth, but that doesn't mean an answer should be altered to be more comfortable but false, or even a straight out lie.
Evaluation: The response attempts to address the user's inquiry by discussing potential measures for improving the helpfulness, ethics, and truthfulness of an AI system like Open Assistant. It mentions the option to flag and label content and emphasizes the importance of truthfulness, despite potential discomfort. However, the response lacks specificity and detailed suggestions for enhancing these properties, making it somewhat incomplete. Additionally, it does not explore potential contradictions between these properties. Overall, while it touches on relevant points, it could provide more comprehensive insights.
Score: 2

---

User: {instruction}
Response: {response}
Evaluation:"""

LLAMA2_PROMPT_TRAIN = """[INST] <<SYS>> <</SYS>>

{instruction} [/INST] {response} </s>"""

LLAMA2_PROMPT_INFER = """[INST] <<SYS>> <</SYS>>

{instruction} [/INST] """

TEMPLATE_MAP = {
    "fewshot": FEWSHOT_TEMPLATE,
    "llm-as-a-judge": JUDGE_TEMPLATE,
    "few-shot-judge": FEWSHOT_JUDGE_TEMPLATE,
    "llama-2-infer": LLAMA2_PROMPT_INFER,
    "llama-2-train": LLAMA2_PROMPT_TRAIN,
}
