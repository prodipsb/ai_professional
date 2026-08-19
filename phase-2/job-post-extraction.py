# =======================================================
# Import the necessary libraries
# =======================================================

from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()

client = Groq()


# ============================================
# SAMPLE MESSY JOB POSTING (realistic, unstructured)
# ============================================
# job_posting = """
# We're hiring! TechNova Solutions is looking for a Senior Backend Engineer 
# to join our growing team. You'll work on our core API infrastructure. 

# Requirements:
# - 5+ years of experience with Node.js or Python
# - Strong knowledge of PostgreSQL and Redis
# - Experience with Docker and AWS
# - Bachelor's degree preferred but not required

# This is a hybrid role based in Dhaka, with 2 days/week in office. 
# Salary range: 80,000 - 120,000 BDT/month depending on experience.
# Apply by sending your resume to careers@technova.example.com
# """

# job_posting = """
# About the job
# Company Name: Impressive Security Limited

# Location: Dhaka (Dhanmondi 27)

# Job Type: Full-Time

# Workplace Type: Full-time

# Job ID: IJOB202654322

# WE'RE HIRING: SENIOR SOFTWARE ENGINEER

# Ticket Lagbe Ltd A Sister Concern of Impressive Group is looking for an experienced Senior Software Engineer to join our growing technology team and help build scalable, secure, and high-performance travel technology solutions.

# Technology: Node.js / NestJS

# Key Responsibilities

# Design and develop scalable back-end applications using Node.js and NestJS
# Build and maintain RESTful APIs and third-party integrations
# Develop flight booking, payment, supplier, and travel-related systems
# Optimize application performance, security, and database queries
# Review code and maintain high development standards
# Troubleshoot production issues and implement reliable solutions
# Guide junior developers and collaborate with frontend, QA, and DevOps teams

# Requirements

# Experience: 5-10 years 
# Education: 
# Skills: travel booking systems, ticketing software, airline ticketing, travel booking, technical hiring, airline reservations, flight test engineering, back-end development, archtics ticketing system, job posting 

# Last Date of Application: 23 August 2026
# """


job_posting = """
About the job
Who We Are:

Field Nation brings companies and service professionals together through an integrated, easy-to-use platform. We support businesses looking to grow their service offerings while also empowering technicians to leverage their skills on their own terms. Our mission is to help the service delivery industry do great work, and we live that mission by doing great work for the companies and service professionals who depend on us.



Why is this role important to Field Nation?

Our Data team is growing its software engineering capability to accelerate the development of internal tools that make our analysts, engineers, and business stakeholders dramatically more effective. We're in the early stages of building AI-powered internal products — including a data analyst chatbot that allows non-technical users to query and explore data through natural language — and we need someone who can bridge the gap between the world of data and modern software engineering.

This is a rare opportunity to shape how an organization uses AI internally: you'll work in a fast-paced, collaborative environment where your opinions matter and your work will be immediately visible to the people who use it every day.



What You'll Get To Do:

Design, build, and maintain internal tools and AI-powered applications that enable data teams and business stakeholders — including chatbots that allow analysts and non-technical users to query and explore data through natural language.
Deploy and operate Python-based applications using Docker and Kubernetes, ensuring reliability and maintainability.
Collaborate with data engineers and analysts to understand workflows and translate them into software solutions that remove friction and multiply impact.
Integrate large language model (LLM) APIs and AI tooling into internal products in thoughtful, practical ways.
Participate in technical design discussions, write specs, and bring fresh ideas about how software and AI can solve real organizational problems.
Contribute to engineering best practices on a data team that is growing its software culture. 


You Might Be a Good Fit If You Have

 Technical Background: 

5+ years of experience in software engineering, with a background that includes data engineering, data analysis, analytics engineering, or data science.
Strong proficiency in Python; comfortable writing clean, tested, production-quality code.
Experience deploying applications using Docker containers and cloud infrastructure (AWS preferred).
Hands-on experience building or maintaining data pipelines, ETL workflows, or similar data infrastructure.
Familiarity with SQL and working with relational or columnar databases.
Experience building lightweight web applications or internal tools — Streamlit, Flask, FastAPI, or similar frameworks.


 AI & Modern Tooling: 

Required: Active, hands-on experience using AI coding assistants and LLM tools such as Claude, GitHub Copilot, Gemini, Cursor, or similar in your day-to-day engineering workflow.
Required Experience (Professional or Personal): Experience with Pydantic AI, creating Agents, function calling, context compression, etc.
Experience working with LLM APIs (Anthropic, OpenAI, etc.) and integrating AI capabilities into applications is a strong plus.
Comfortable evaluating AI-generated code critically and iterating quickly with AI assistance.


 Collaboration & Communication: 

Ability to work closely with non-engineers (analysts, product managers, business stakeholders) to understand needs and translate them into software solutions.
Excellent written and verbal communication skills.
Experience or interest in helping a data-centric team grow its software engineering culture.


Nice To Have:

Prior experience building internal tooling, data products, or analyst-facing applications.
Experience with LLM orchestration frameworks such as LangChain, LlamaIndex, or similar.
Familiarity with dbt, Airflow, Spark, or other data engineering tools.
Experience with vector databases or retrieval-augmented generation (RAG) architectures.
Background in analytics engineering or working closely with BI tools (Tableau, Looker, etc.).
 

Education:

B.S. or M.S. in Computer Science, Data Science, Engineering, or a related field — or equivalent practical experience. We care more about what you've built than where you studied.


Location and Work Schedule: 

Working Days: Hybrid Monday to Friday (3 days from Home + 2 days from the Office)
Working Hours: 1:00 PM to 10:00 PM
Location: Uttara 12, Dhaka, Bangladesh


Salary Range:

BDT 160,000 - 210,000 (Monthly) 


Why We Think You'll Love It Here:

At Field Nation, we believe great work deserves great support. Here’s a snapshot of the benefits designed to reward performance, support well-being and create an enjoyable workplace experience.

 

1. Compensation & Financial Rewards 

Because hard work should feel like winning.

Field Nation LLC Performance Reward – Because every citizen of Field Nation deserves a stake in the win!
Festival Bonus – Celebrate the big festivals with some extra cheer (and cash!).
Referral Bonus – Incentives for successful employee referrals.
Gratuity – Honoring your long-term dedication
Leave Encashment – Opportunity to encash unused annual leave balance at year-end.


2. Health & Wellness

Your body, mind, and family matter.

Medical Insurance – Comprehensive health coverage for employees and their immediate family (spouse and children).
Gym Membership – Stay fit, active, and energized. 


3. Daily Operations & Office Perks

Complimentary Lunch / Dinner – Because good work needs good food.
Unlimited Tea & Coffee – Keep the energy flowing.
Transportation – Helping you get to work hassle-free.
Mobile Data Allowance – Allowances to ensure connectivity. 


4. Professional Development

Career Development Budget – Dedicated funds for professional learning and growth. 


5. Culture, Events & Time Off

Fostering work-life balance and strong team connections. 



Work Model: Hybrid (2 days in-office, 3 days remote per week) – balance is key.
Summer & Winter Field Weeks – Two annual team retreats to connect, collaborate, and recharge.
Quarterly Team Outing Budget – Enjoy exciting activities and quality time with your team to bond, relax and celebrate together.
Occasional Gifts – Surprises and gifts to celebrate milestones & welcome new faces.
Leave Benefits:
Maternity Leave
Paternity Leave
Hajj/Umrah Leave
Paid Time Off – Take the time you need! Covers annual, casual, and sick leave so you can recharge and come back ready to shine. 


Why Field Nation?

At Field Nation, we are breaking the barriers to work and enabling the gig economy. We’re a tech company that offers a web-based marketplace solution for buyers and sellers of contract services to simply connect, work, and manage their business. We look for collaborators, innovators, and problem solvers to join us in our common purpose of changing the way work gets done. We were named a Top Workplace by the Star Tribune in 2017, 2018, 2019, and a Top 150 Workplace in 2020. We look to hire extraordinary people and provide them extraordinary benefits.

We may use artificial intelligence (AI) tools to support parts of the hiring process, such as reviewing applications, analyzing resumes, or assessing responses and identifying potential inconsistencies or verification signals in application materials based on available information. These tools assist our recruitment team but do not replace human judgment. Final hiring decisions are ultimately made by humans. If you would like more information about how your data is processed, please contact us.
"""



 # =======================================================
 # TECHNIQUE 1: STRUCTURE INSTRUCTIONS
 # =======================================================

 # Instand of a vague ask, we give the LLM:
 #  - A clear ROLE ("You are a ...")
 #  - A clear TASK (What exactly to do)
 #  - Clear CONSTRAINTS (What format, what to do if data is missing)
 #  - An explicit OUTPUT FORMAT (So parsing it in code is reliable)


def build_prompt(job_text):
    prompt = f"""You are a precise data extraction assistant. Your job is to extract structured information from job postings.
TASK: Extract the following fields from the job posting below.Groq

RULES:
- If a field is not mentioned in the text, use null (not "N/A" or empty string)
- years_of_experience should be a number only (e.g. 5), not text
- required_skills should be a list of individual skill strings
- Response with ONLY valid JSON, no explanation, no markdown code blocks

OUTPUT FORMAT (follow this exact structure):
{{
    "job_title": string,
    "company": string,
    "years_of_experience": number or null,
    "required_skills": [list of strings],
    "salary_range": string or null,
    "work_mode": "remote" | "hybrid" | "onsite" | null,
    "location": string or null
}}


JOB POSTING:
{job_text}


JSON OUTPUT:

"""
    return prompt



# =========================================================
# CALL THE LLM
# =========================================================

# def extract_job_info(job_text):

#     prompt = build_prompt(job_text)

#     response = client.chat.completions.create(
#         model="openai/gpt-oss-20b",
#         max_tokens=400,
#         temperature=0,
#         messages=[{"role": "user", "content": prompt}]
#     )

#     raw_output = response.choices[0].message.content

#     # The LLM sometimes wraps JSON in markdown code blocks (```json...```)
#     # even when told not to. This cleans that up just in case.
#     cleared = raw_output.strip().removeprefix("```json").removeprefix("```").removeprefix("```").strip()

#     # Convert the JSON STRING into an actual Python dictionary
#     # so we can use it programmatically ( not just print it as text)
#     try:
#         parsed = json.loads(cleared)
#         return parsed
#     except json.JSONDecodeError as e:
#         print(f"Failed to parse JSON: {e}")
#         print(f"Raw output was: {raw_output}")
#         return None



def extract_job_info(job_text):
    prompt = build_prompt(job_text)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=1200,          # increased, just in case
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    # NEW: Print diagnostic info BEFORE trying to parse anything
    choice = response.choices[0]
    print(f"DEBUG - finish_reason: {choice.finish_reason}")
    print(f"DEBUG - raw content length: {len(choice.message.content or '')}")
    print(f"DEBUG - raw content: {repr(choice.message.content)}")
    print("-" * 60)

    raw_output = choice.message.content

    if not raw_output:
        print("ERROR: LLM returned empty content!")
        return None

    cleaned = raw_output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw output was: {raw_output}")
        return None


# ==================================================
# RUN IT
# ==================================================

result = extract_job_info(job_posting)

print("Extracted structured data:")
print(json.dumps(result, indent=2))     # pretty-print the dictionary


# Prove it's REAL structured data, not just text - access individual fields

if result:
    print(f"\nJust the skills: {result['required_skills']}")
    print(f"Just the experience requirement: {result['years_of_experience']} years") 

