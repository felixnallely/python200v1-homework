import nest_asyncio
nest_asyncio.apply()
print("nest_asyncio applied")

from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
# --- Concepts Question 1 --- # 
# Scenario A:
#   RAG. Since the legal team need answers from hundreds of PDFs that change every quarter. So RAG is the best way to do this
#   because it allows for information to be retrieved without retraining the model when everything is updated.

# Scenario B:
#   Fine-tuning. Since the startup wants a unique brand voice and have 3,000 examples of the type of style they want. It is best 
#   to use fine-tuning because it teaches the model the specific writing model the company wants and it will be easier to get the voice they want.

# Scenario C:
#   Prompt Engineering. Since the analyst only wants to ask questions about this specific 2 page report the model does not need to be reusable. Therefore 
#   prompt engineering should be used since it can include only the report in the prompt. 


# --- Concepts Question 2 --- # 
#Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real situation where a confident hallucination could cause harm.
#   It can be more harmful because an "I am not sure" can lead to a fact check but when it's not admitted it's easier to trust. Wrong answers can be put to use in real life environments it can be very dangerous and harmful in areas such as healthcare and law enforcement. 
#   For example when someone asks Chatgbt whether or not they should drink bleach. There are many similar cases when AI responded in a way that lead to harm rather than just answering with "I am not sure."
#   If a response tends to sound more friendly, confident and honest we are likely to trust it and less likely to fact check anything. 


# --- Concepts Question 3 --- # 
#steps = [
    #"Generate a response from the LLM",
    #"Extract text from source documents",
   # "Receive the user's query",
    #"Retrieve the most relevant chunks",
    #"Convert text chunks into embeddings",
   # "Inject retrieved chunks into the prompt",
    #"Split text into chunks",
    #"Embed the user's query", ]


#comment: Rearrange add a one-sentence description of what happens at each step.
# steps = [
#     "Extract text from source documents",  #The system reads the PDFs, web pages, documents.
#     "Split text into chunks",   #Large documents split and broken down into smaller pieces that are easier to go over.
#     "Convert text chunks into the prompt", #Chuncks are transformed to numerical representation anf stored in databases (vector).
#     "Receive the user's query",  #The user asks a question. 
#     "Embed the user's query",  #Question gets converted to embedding (numerical representation) to compare to the stored chunks.
#     "Retrieve the most relevant chunks",  #Finding the chunck relevant with the question. 
#     "Inject retrieved chunks into the prompt",  #Relevant chunks get added to the prompt and sent to LLM.
#     "Generate a response from the LLM",  #Models usese retrieved information and user's question to form an answer. 

# --- KEYWORD RAG --- # 
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# --- Keyword Question 1 --- # 
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)

print("Selected document:", results[0][0])


# comment explaining which document was selected and why.
#   loyalty.txt was selceted because it shared the keyword "your." 
#   hours.txt also contains the keyword "weekends" but both overlapped and loyalty.txt was returned first. 


# --- Keyword Question 2 --- # 
query = "Do you have anything without caffeine?"

results = simple_keyword_retrieval(query, documents, verbose=True)

print("Selected document:", results[0][0])

#comment explaining:

#Which document was selected
#   Selected document: None found
#Whether keyword RAG got this right — and why or why not
#   Yes RAG got this correct because the document doesn't contain the word "caffeine." And since the word does not match
#   the exact word it can't understand the question is about caffeine free drinks. 
#What kind of retrieval would do better here
#   Semantic retrieval because it would be able to recognize the question is about caffeine free drinks. 


# --- Keyword Question 3 --- # 
query = "How do I sign up for rewards?"

results = simple_keyword_retrieval(query, documents, verbose=True)

print("Selected document:", results[0][0])


#Was your prediction correct? If the result surprised you, add a comment explaining what happened.
#   Prediction: none found.


# --- Semantic Question 1 --- # 
#What is a vector embedding? (1-2 sentences)
#   Turning words/text into a list of numbers that represents them. 
#Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more relevant, and what does that number tell you about the relationship between the texts? 
#   0.85 is more relevant since it's a higher score and it tells me that the embedding is closer to the query embedding. So this means the texts are more similar to each other.  
#Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
#   Because semantic search doesn't focus on exact words only on meaning. 

# --- Semantic Question 2 --- # 
#| Feature                    | Keyword RAG                       | Semantic RAG                        |
#|----------------------------|-----------------------------------|-------------------------------------|
#| What is compared?          | Exact word overlap                | Vector embedding                    |
#| What is retrieved?         | Full document                     | Relevant text chunks                |
#| Can it handle synonyms?    | No                                | Yes                                 |
#| Storage format             | Plain text dictionary             | Vector database / embedding index   |
#| Relevance score            | Number of overlapping keywords    | Cosine similarity score             |


# --- LlamaIndex --- # 
from pathlib import Path
from pypdf import PdfReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding 
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator 

load_dotenv()

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader("brightleaf_pdfs").load_data()

print(f"Loaded {len(docs)}documents")
print(docs[0].text[:500])

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs) 

query_engine = index.as_query_engine(similarity_top_k=3) 


response = query_engine.query("What services does BrighLeaf provide?")

print(response)

# --- LlamaIndex Question 1 --- # 
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
    "Should my friend work for BrighLeaf?", ##vague question for LlamaIndex question 3
]
for question in questions:
    print("\n" + "=" * 80)
    print("Question:", question) 

    response = query_engine.query(question)

    print("\nAnswer:", response)
    print("\nSOURCE NODES:")

    for i, node in enumerate(response.source_nodes, start=1):
        print(f"\nNode {i}")
        print(f"Similarity Score: {node.score:.4f}")

        chunk = node.node.text[:150].replace("\n", " ")
        print(f"Chunk: {chunk}...")

#Do the retrieved chunks look relevant to the question?
#  Query 1: Yes, they look relevant and focus on security and what the questions ask.
#  Query 2: Yes, the chunks were relevant. There was a chunk with the highest score that matched the question on network and data security policies.

#Does the model's response sound confident and specific, or does it hedge with phrases like "based on the context" or "I'm not sure"? Note what you observe about the tone.
#   Query 1: Yes, the response sounds confident and specific since it is very detailed.
#   Query 2: Yes, the responses were confident and specific. As well as detailed. 

#Did anything unexpected get retrieved?
#   Query 1: This part "EcoVolt Energy (2022 Partnership) BrightLeaf's collaboration" was irrelevant to employee benefits. 
#   Query 2: There was a chunk about employee benefits. 

# --- LlamaIndex Question 2 --- # 
#With similarty_top_k=1:
# BrightLeaf offers a comprehensive benefits program that includes:
#1. **Health, Vision, and Wellness**: Medical insurance through Blue Cross, covering preventive care, hospitalization, telemedicine, and mental health services. Vision benefits include annual exams and 
# allowances for lenses and frames. Employees can participate in wellness programs and utilize a Wellness Reimbursement Plan forup to $600 annually for health-related activities.

# With similarty_top_k=5:
# Answer: BrightLeaf offers a comprehensive benefits program that includes health, vision, and wellness support, financial security and retirement plans, parental leave, work flexibility, and opportunities for 
# learning and career growth.

#comment explaining how the response changed (if at all) and whether more retrieved context is always better.
#   The similarty_top_k=1 provided a shorther response that focused on health, vision, and wellness. But the similarty_top_k=5 was comprehensive and included more information that the first k=1 did not. 
#   Having more retrieved context is not always better. Although it can help gve more information but often it can be unrelated to the question being asked. 


# --- LlamaIndex Question 3 --- # 

#explaining what you expected
#   I expect the model to respond with "I am not sure."

# what actually happened? 
#   The model answered the question and even provided why it would be a good work environment. But still the model didn't
#   directly answer the question it just provided information about BrightLeaf. 

# what you would change about the system to handle this kind of query better.
#   I would add a prompt that tells the model to state when an answer isn't supported instead of trying to answer it anyways. 


# --- LlamaIndex Question 4 --- # 
import inspect 

judge_llm = OpenAI(model="gpt-4o-mini")

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)

#debugging since I kept getting an error
#print("Faithfulness evaluate_response:")
#print(inspect.signature(faithfulness_evaluator.evaluate_response))

relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

#Good query
q1 = "What employee benefits does BrightLeaf offer?"

response1 = query_engine.query(q1)

contexts = [node.node.text for node in response1.source_nodes]

faithfulness1 = faithfulness_evaluator.evaluate(query=q1, response=str(response1), contexts=contexts)

relevancy1 = relevancy_evaluator.evaluate(query=q1, response=str(response1), contexts=contexts)

print("\nGOOD QUERY")
print("Question:", q1)
print("Faithfulness:", faithfulness1)
print("Relevancy:", relevancy1)

#Weak query 
q2 = "Who will be World Champion of Formula One in 2027?"

response2 = query_engine.query(q2)

contexts = [node.node.text for node in response2.source_nodes]

faithfulness2 = faithfulness_evaluator.evaluate_response(query=q2, response=str(response2), contexts=contexts)

relevancy2 = relevancy_evaluator.evaluate(query=q2, response=str(response2), contexts=contexts)


print("\nWEAK QUERY")
print("Question:", q2)
print("Faithfulness:", faithfulness2)
print("Relevancy:", relevancy2)

#----**
# FaithfulnessEvaluator and RelevancyEvaluator raise an async runtime error in Python 3.14.3, llmax-index 0.14.26, openai 3.51.0.

#----**
# But 
#What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
#   A faithfulness score of 1.0 would mean the answer is fully fupported by the documents the model was provided and that no information was made up. 
#   A score of 0.0 would mean the answer was made up and not supported by the documents. 

#What does a relevancy score measure, and how is it different from faithfulness?
#   The relevancy score measures how well the user's question was answered by the response. While the faithfulness score measures how well supported the response was by the documents the model has.

#Did the scores change between your two queries? If so, why do you think that happened?
#   Yes because the employee benefits query would have higher scores in both faithfulness and relevancy because the information is in the BrightLeaf documents. But the Formula 1 question would have very low scores
#   since there are no documents to support a relevant answer. 

#What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?
#   It's a useful RAG becawuse of its qualities of faithfulness, and relevance aren't easy to measure with an accuracy score. It also needs to examine the context and determine if the documents support the answers to the question. 