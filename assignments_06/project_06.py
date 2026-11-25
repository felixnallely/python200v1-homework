from pathlib import Path 
from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Settings 
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Step 1: Set Up --- # 
load_dotenv()

if load_dotenv():
    print("API key loaded successfully.")
else: 
    print("Warning: could not load API key.")


docs_dir = Path("groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print("Document folder found.")


# ---   Step 2: Load the Documents --- # 
documents = SimpleDirectoryReader(docs_dir).load_data()

print(f"Loaded {len(documents)} documents.")

for doc in documents: 
    print("-", doc.metadata["file_name"])


# --- Step 3: Build the Index --- # 
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)

print("Index built successfully. Ready to answer questions.")

# --- Step 4: Query the Assistant --- # 
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions: 
    print("\n" + "=" * 80)
    print("Questions:", question)

    response = query_engine.query(question)

    print("\nAnswer:", response)

    if response.source_nodes:
        top_node = response.source_nodes[0]
        print("\nTop Retrieved Source Node:")
        print("Document:", top_node.node.metadata.get("file_name", "Unknown"))
        print("Similarity Score:", round(top_node.score, 4))

        chunk = top_node.node.text[:200].replace("\n", " ")
        print("Chunk Preview:", chunk)

#Did the assistant sound confident and accurate? Did any of the answers surprise you?
#   Yes, the assistant did sound confident and accurate. What suprised me was how the model answered the questions and was able to make the answers flow
#   without making them hard to understand and repeating what was written in the documents. 


# --- Step 5: Find a Failure --- # 
difficult_question = "What is the most popular drink?"

response = query_engine.query(difficult_question)

print("\n" + "=" * 80)
print("Difficult Question:", difficult_question)

print("\nAnswer:", response)

print("\nRETRIVED SOURCE NODES:")

for i, node in enumerate(response.source_nodes, start=1):
    print(f"\nNode {i}")
    print("Documents:", node.node.metadata.get("file_name", "Unknown"))
    print("Similarity Score:", round(top_node.score, 4))
    
    chunk = top_node.node.text[:200].replace("\n", " ")
    print("Chunk Preview:", chunk)

# What you asked and why you expected it to be hard
#   I asked, "What is the most popular drink?" I expected it to be difficult because the documents contain iformation on the drinks and specials but it doesn't contain any rankings or the most purchased or popular drink.

# What went wrong — wrong retrieval, missing information, the model guessed anyway?
#   The model answered as: "The information provided does not specify which drink is the most popular." Therefore nothing went wrong since the model admitted it has no information on this topic. 

# When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound confident even when it was wrong? What does this suggest about trusting AI-generated responses?
#   Yes, it sounded less confident since I didn't know how to answer. 

# What you would change about the system to improve it
#   I would add more documents that show data sales, and customer preferences so the model would be able to answer questions like these. 


# --- Step 6: Reflection --- #

#The LlamaIndex lesson lists the steps that LlamaIndex automates: chunking, embedding, and indexing. How many lines of code did your LlamaIndex implementation take in your project? Compare that with the keyword RAG code from the Keyword-based RAG lesson, or with the code you would need to write each of those steps yourself. What does that tell you about the value of using a framework?
#   My LlamaIndex implementation took a few lines of code while the keyword RAG required more custome code to be able to process everything. This tells me that a framework reduces development time which in helps focus more time to focus on problem solving instead of doing everything from scratch. 
#   

#You have now built a system that answers questions from real documents. Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.
#   Applying this system in healthcare would help gain actual value. Since a model like this can cut down on time when searching a patients information as well as staff members. It would make it easier to keep up with schedueling and get needed information such as treatment guidelines. 

#What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?
#   RAG can't prevent problems that are caused by outdated, incomplete or wrong documents. The model will just use the information it has to give an answer even if the answer is wrong. 