# ==================================================
# Import libraries
# ==================================================

from pydantic import BaseModel, Field
from typing import Optional, Literal
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()

client = Groq()


# ==================================================
# Define the schema using Pydantic
# ==================================================

class JobExtraction(BaseModel):

    # Field(...) lets us add extra info like descriptions,
    # which Pydantic can later use to auto-generate the JSON schema for us

    job_title: Optional[str] = Field(None, description="The job title")
    company: Optional[str] = Field(None, description="The hiring company name")
    years_of_experience: Optional[int] = Field(None, description="Required years of experience, as a number")
    required_skills: list[str] = Field(default_factory=list, description="List of technical skills required")
    salary_range: Optional[str] = Field(None, description="Salary range in mentioned")

    # Literal restricts this field to ONLY these 3 exact values -
    # if the LLM tries to return anything else, Pydantic will REJECT it
    work_mode: Optional[Literal["remote", "hybrid", "onsite"]] = Field(None, description="Work arrangement")
    location: Optional[str] = Field(None, description="Job location")


# ========================================================
# AUTO GENERATE THE JSON SCHEMA FROM OUR PYDANTIC MODEL
# ========================================================

# Instead of manually writing out the JSON structure in our prompt 
# (Like we did in Lesson 1), Pydantic can generate it for us -
# guaranteeing the prompt and our validation always stay in sync.

# schema = JobExtraction.model_json_schema()
# print("Auto-generated schema")
# print(json.dumps(schema, indent=2))



# ========================================================
# BUILD THE PROMPT USING THE AUTO-GENERATED SCHEMA
# ========================================================

def build_prompt(job_text):
    schema = JobExtraction.model_json_schema()

    prompt = f"""You are a precise data extraction assistant.

TASK: Extract structured information from the job posting below.

RULES:
- required_skills should include ALL technical skills mentioned anywhere in the position, listed ONCE each (no duplicates)
- Keep the list concise - extract distinct technologies/tools only, not every possible phrasing


JSON SCHEMA:
{json.dumps(schema, indent=2)}

JOB POSTING:
{job_text}

JSON OUTPUT:

"""
    return prompt

# ========================================================
# BUILD THE PROMPT USING THE AUTO-GENERATED SCHEMA END
# ========================================================



# ========================================================
# CALL THE LLM AND VALIDATE WITH PYDANTIC
# ========================================================

def extact_job_info(job_text):

    prompt = build_prompt(job_text)

    # print(f"build prompt: {prompt}")

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        max_tokens=2000,
        temperature=0,
        messages=[{ "role": "user", "content": prompt }]
    )


    # print(f"response: {response}")

    # DEBUG - add this before anything else
    choice = response.choices[0]
    print(f"DEBUG - finish_reason: {choice.finish_reason}")
    print(f"DEBUG - raw content: {repr(choice.message.content)}")
    print("-" * 60)

    raw_output = response.choices[0].message.content

    print(f"LLM response: {raw_output}")

    cleaned = raw_output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        # Parse the raw JSON text info a Python dict first
        data = json.loads(cleaned)

        # This is the key new step:
        # JobExtraction(**data) validates the dict against our Pydantic schema.
        # If ANY field has the wrong type, or work_mode isn't exactly
        # "remote"/"hybrid"/"onsite", this line will RAISE AN ERROR immediately
        # - instend of letting bad data silently flow into your application.

        validated = JobExtraction(**data)

        return validated
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return None
    except Exception as e:            # catches Pydantic ValidationError specifically
        print(f"Schema validation failed: {e}")
        print(f"Raw data was: {data}")
        return None


# ===================================================
# TEST IT
# ===================================================

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


result = extact_job_info(job_posting)

if result:
    print("Validated result (as Pydantic object):")
    print(result)
    print()


    # New: notice how clean attribute access is now - no more result['job_title],
    # Just result.job_title - with autocomplete support in your editor too

    print(f"Job Title: {result.job_title}")
    print(f"Skills: {result.required_skills}")
    print()

    # BONUS: convert back to a clean dict/JSON anytime you need it
    print("As JSON:")
    print(result.model_dump_json(indent=2))



