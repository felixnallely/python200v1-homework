from dotenv import load_dotenv
from openai import OpenAI
import json

# --- Task 1: Setup and System Prompt --- #
load_dotenv()
client = OpenAI()

def get_completion(
        messages, 
        model="gpt-4o-mini", 
        temperature=0,
        max_completion_tokens=400
):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,

        max_completion_tokens=max_completion_tokens
    )
    return response.choices[0].message.content

system_prompt = """
You are an experienced job application coach who helps job seekers create and improve
resumes, cover letters, professional summaries, interview responses, and other job search materials.

Provide constructive, professional, and practical feedback.

Stay focused on career-related communication and job application materials. 

When creating the content make sure to encourage users to review, edit, and personalize your suggestions 
before submitting them to future employers.

Acknowledge that hiring expectations are different across industries and regions. Mention that you may not know 
the specific norms for the user's field. So the user should apply their own judgment and decide whether or not to use the recommendations.
"""

#comment explaining at least one deliberate choice you made in writing the system prompt and why.
#   I limited how the model response by telling it to give professional and pratical feedback. I did this so the tone and responses stay career relevant. 


# --- Task 2: Bullet Point Rewriter --- # 
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. 
    
    Each item should have two keys:
    "original" (the original bullet) and 
    "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = get_completion(messages)

    response = response.replace("```json", "")
    response = response.replace("```", "")
    response = response.strip()

    start = response.find("[")
    end = response.rfind("]") + 1

    response = response[start:end]

    results = json.loads(response)

    for item in results:
        print("Original:", item["original"])
        print("Improved:", item["improved"])

    return results 

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

#a comment: What makes these bullets weak, and what kinds of changes did the model suggest?
#   These bullets are weak because they are vague and don't give clear results. The model made the bullets 
#   stronger by using strong action verbs and professional language. 

# --- Task 3: Cover Letter Generator --- # 
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    
    Role: {job_title}
    
    Background: {background}
    
    Opening:
    """

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = get_completion(messages)

    return response 

job_title = "Junior Data Engineer"

background = (
    "Five years of experience as a middle school math teacher; recently completed \ a Python course and built data pipelines using Pandas."
)

letter = generate_cover_letter(job_title, background)

print("Letter:")
print(letter)


#comment: Why did you choose those particular examples? What does the few-shot pattern help control in the output?
#   I kept these examples as is because the prompt focused on career changes. The examples kept a specific, confidence and technical skills. 
#   The few-shot prompting helps control structure, style and details in the output so the response is more tailored.


# --- Task 4: Moderation Check -- # 
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    #print("Category Triggered")
    #print(result.results[0].categories)

    flagged = result.results[0].flagged

    if flagged: 
        print("Your message may contain bad content. Please rephrase and try again.")
        return False 
    return True 

#Test 1
safe_text = "Can you please help me improve my resumer for a position in data analyst?"
print("Safe Text:")
print(is_safe(safe_text))

#Test 2 
flagged_text = "How can I do something illegal and not get caught?"
print("Flagged Text:")
print(is_safe(flagged_text))


# --- Task 5: The Chatbot Loop --- #
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            if raw_bullets: 
                rewrite_bullets(raw_bullets)

            print()

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            
            letter = generate_cover_letter(job_title, background)
            print("\nCover Letter Opening: \n")
            print(letter)


        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append(
                {"role": "user", "content": user_input
                }
            )

            reply = get_completion(messages)
            print("\nJob Application Helper:")
            print(reply)

            messages.append(
                {"role": "assistant", "content": reply
                }
            )

if __name__ == "__main__":
    run_chatbot()


# --- Task 6: Ethics Reflection --- # 
# 1. Your bot was trained on text written by and about certain kinds of people. How might this produce biased advice? Could it favor certain communication styles, industries, or cultural backgrounds?
#   Yes, since my bot was trained on certain communication styles and industries it might lack more knowledge in other regions, clutures and industries. Therefore my bot could give biased responses towards data enginneering 
#   and give negative or harmful help on teaching. The bot can also recommend a certain communication style for one environment without affecting an industry, but it can give a wrong communication style that can be seen as a problem in another cultural industry.

# 2. What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?
#   Without out checking whether the tone and style fits the person can make the person sound robotic. This can cause an employeer to be disappointed. This can also make the job-sheeker lose their 
#   credibility, and can even cause the job-seeker to be blacklisted. Another thing is that when the bot makes a mistake and the mistakes are submitted to the employer directly, the job-seeker won't be represented 
#   with accurate skills and experience. 

# 3. What is one guardrail you would add if you were deploying this tool professionally? (A guardrail is any design choice that reduces the chance of harm — a UI warning, a moderation filter, a usage policy, a disclaimer, or something else entirely.)
#   I would add an extra warning reminding users to check double check and edit any feedback they recieved from the bot. I would also add a disclaimer that would state that the bot is giving suggestions not professional advice. 
#   Adding these two guardrail will help the user review the responses given by the bot so that errors will be reduced and this will tell the user they are responsible for how they use the information they recieved. 