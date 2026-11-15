from dotenv import load_dotenv
from openai import OpenAI

if load_dotenv():
    print("Loaded environment variables from .env file")
else: 
    print("Warning: .env file not found.")

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Total Tokens:", response.usage.total_tokens)


# --- API Question 2 --- # 
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures: 
    response = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{"role": "user", "content": prompt}],
                                              temperature=temp)
    print(f"\nTemperature: {temp}")
    print("Response:", response.choices[0].message.content)

#What do you notice about how the outputs differ? 
#   I noticed that temperature 0 and 1.5 resultted in the same response. But temperature tends to control the randomness of the outputs. 
#   Though lower temperatures tend togive predictable responses while higher temperatures result in more creative responses. 
#Which temperature would you use if you needed a consistent, reproducible output?
#   I would use temperature = 0, because it minimizes the randomness. 


# --- API Question 3 --- # 
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices, start=1):
    print(f"Completion {i}:")
    print(choice.message.content)
    print()


# --- API Question 4 --- # 
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

print("Response:", response.choices[0].message.content)


#comment: What happened, and why might you want to use max_tokens in a real application?
#   The response was cut short. Normally the response would be much longer but since the max_tokens was limited to 15 it reduced the response length. 
#   In a real application using max_tokens can help reduce API cost, helps control reponse length and can also help improve response speed.


# --- System Question 1 --- #
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages=messages
)

print("Tutor Response:", response.choices[0].message.content)


messages = [
    {"role": "system", "content": "You are a motivational coach."},
    {"role": "user", "content": "I don't understand python, I should give up."}
]

response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages=messages
)

print("Motivational Response:", response.choices[0].message.content)

#comment noting what changed.
#   The answer is different for each question and they also have a different tone and its wording were different. 


# --- System Question 2 --- #
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)

# comment: Why does the model know Jordan's name, even though it's stateless?
#   The model knows Jordan's name because the entire conversation history was included in the message list and was sent to the API.


# --- Prompt Question 1 — Zero-Shot --- # 
print("\n--- Q1: Zero-Shot Prompting ---")

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""Classify the sentiment of each review as positive, negative, or mixed.
            
            Review:{review}"""
            }
            ]  
    )

    print(f"Review {i}: {response.choices[0].message.content}")

# --- Prompt Question 2 — One-Shot --- # 
print("\n--- Q2: One-Shot Prompting ---")

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""Classify the sentiment of each review as positive, negative, or mixed.
                
                Example:
                Review: "Fast shipping but the item arrived damaged."
                Sentiment: mixed

                Review:"{review}"
                Sentiment:
                """
                }
                ]  
        )
    
    print(f"Review {i}: {response.choices[0].message.content}")

#comment: Did adding one example change the format or consistency of the output compared to Q1?
#   Yes, adding one example helps the model understand what kind of answer to give and whether or not 
#   to write that answer in a full sentence. 


# --- Prompt Question 3 — Few-Shot --- # 
print("\n--- Q3: Few-Shot Prompting ---")

for i, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""Classify the sentiment of each review as positive, negative, or mixed.
                
                Example:
                Review: "The staff was helpful and the service was excellent."
                Sentiment: positive

                Example: 
                Review: "The app is buggy and crashes often."
                Sentiment: negative

                Example:
                Review: "The design looks amazing, but creating it will be time consuming."
                Sentiment: mixed

                Review:"{review}"
                Sentiment:
                """
                }
                ]  
        )
    
    print(f"Review {i}: {response.choices[0].message.content}")

#comment comparing all three approaches (zero-shot, one-shot, few-shot): When would you choose each one?
#   Zero-shot: When instructions are clear and no output format structure is needed.
#   One-shot: when an example and expected format or behavior is needed. 
#   Few-shot: used when consistent results, tasks and output format is important.



# --- Prompt Question 4 — Chain of Thought --- # 
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content":"""
        Solve the following problem. Show your reasoning step by step before giving the final answer.
        Label the final answer clearly.
        
        A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
        takes a new job that pays $7,500 more per year than her post-raise salary.
        What is her final annual salary?
        """}
    ]
)

print(response.choices[0].message.content)

#comment: Why does asking the model to reason step by step tend to improve accuracy on problems like this?
#   Because it forces the model to work through the problem through each part instead of making it directly give the answer.
#   Doing this helps reduce errors and makes the logic easy to correct and follow. 


# --- Prompt Question 5 — Structured Output --- # 
import json 

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": f"""
        Analyze the following review.
        
        Return ONLY valid JSON with these keys:
        - sentiment
        - confidence
        - reason

        confidence should be a float from 0 to 1. 

        reason should be one sentence. 

        Review:"{review}"
        """
        }
    ]
)

raw_response = response.choices[0].message.content

print("Raw Response:", {raw_response})

try: 
    result = json.loads(raw_response)

    print("\nParsed Fields:")
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except json.JSONDecodeError:
    print("\nERROR: Response was not valid JSON.")
    print("Raw response for debugging:", {raw_response})


# --- Prompt Question 6 — Delimiters --- # 
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.responses.create(
    model="gpt-4o-mini",
    input=prompt
)

print(response.output_text)


user_text = "The moon was bright. The smell of spring was in the air."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.responses.create(
    model="gpt-4o-mini",
    input=prompt
)

print(response.output_text)

#comment: What problem do delimiters help prevent?
#   They help prevent prompt injection and keep result predictible. 


# --- Ollama Question 1 --- # 
from openai import OpenAI 

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain what a large language model is in two sentences."
         }
    ]
)

print(response.choices[0].message.content)

"""
Ollama Output:

A large language model is an AI system designed to understand and generate 
human-like text, enabling tasks like writing, answering questions, or 
creating content. It is trained on vast datasets, allowing it to learn 
patterns and understand context, making it highly efficient for various 
applications.

"""
"""
OpenAI output:
A large language model is an advanced artificial intelligence system designed to 
understand and generate human language by processing vast amounts of text data. 
It uses complex algorithms and deep learning techniques to predict the likelihood 
of word sequences, enabling it to respond to prompts, answer questions, and engage 
in conversation in a coherent and contextually relevant manner.
"""


#multi-line string comment in your code. 
#Then add another comment answering: What differences did you notice between the two responses? What is one advantage 
#and one disadvantage of running a model locally?
#   Differences: 
#   The OpenAI has a detailed and polished response while the Ollama response is shorter and more concise. 
#   An advantage of running locally is there is better privacy when data stays on a personal computer. A disadvantage
#   is that this requires local hardware resournces and can also produce lower quality responses. 





