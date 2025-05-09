import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from openai import OpenAI
import re

from dotenv import load_dotenv


load_dotenv() 

token = os.environ.get("GITHUB_TOKEN") 
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

analyzer = SentimentIntensityAnalyzer()

st.title("🤖 AI Interview Practice Chatbot")

# ---- USER INPUT ----
job_role = st.text_input("Enter the job role you're interviewing for:", "Data Scientist")

# ---- SESSION STATE ----
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "user_answers" not in st.session_state:
    st.session_state.user_answers = []
if "feedback_shown" not in st.session_state:
    st.session_state.feedback_shown = False

# ---- GENERATE QUESTIONS ----
def generate_questions(role):
    prompt = (
        f"Generate 3 behavioral and 2 technical interview questions for a {role} role. "
        "Please list only the questions, numbered."
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
    )
    content = response.choices[0].message.content.strip()
    lines = content.split("\n")
    questions = [line.strip() for line in lines if line.strip() and "?" in line]
    return questions

if st.button("Start Interview"):
    st.session_state.questions = generate_questions(job_role)
    st.session_state.current_q = 0
    st.session_state.user_answers = []
    st.session_state.feedback_shown = False

# ---- INTERVIEW CHAT FLOW ----
if st.session_state.questions:
    q_idx = st.session_state.current_q
    if q_idx < len(st.session_state.questions):
        st.subheader(f"Question {q_idx + 1}")
        st.write(st.session_state.questions[q_idx])
        answer = st.text_area("Your answer:", key=f"answer_{q_idx}")

        if not st.session_state.feedback_shown:
            if st.button("Submit Answer"):
                st.session_state.user_answers.append(answer)
                sentiment = analyzer.polarity_scores(answer)

                feedback_prompt = (
                    f"Evaluate how well the following answer responds to the interview question "
                    f"in terms of relevance, completeness, and clarity.\n\n"
                    f"Question: {st.session_state.questions[q_idx]}\n"
                    f"Answer: {answer}\n\n"
                    f"Provide detailed feedback, then add a score out of 10 using this format:\n"
                    f"Rating: X/10"
                )

                feedback_response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": feedback_prompt}],
                    temperature=0.7,
                    top_p=1.0,
                    max_tokens=500,
                )

                feedback_text = feedback_response.choices[0].message.content.strip()

                # --- SAFELY EXTRACT RATING ---
                rating = None
                if "Rating:" in feedback_text:
                    rating_line = [line for line in feedback_text.split('\n') if "Rating:" in line]
                    if rating_line:
                        rating_str = rating_line[0].split(":")[1].strip()
                        match = re.search(r'\d+', rating_str)
                        if match:
                            extracted = int(match.group())
                            if 0 <= extracted <= 10:
                                rating = extracted
                            else:
                                st.warning("⚠️ The rating extracted is outside the valid range (0–10).")
                        else:
                            st.warning("⚠️ Could not extract a valid numeric rating from the feedback.")

                st.markdown("### 🧠 Feedback")
                st.write(feedback_text)

                if rating is not None:
                    st.markdown("### 📊 Rating Progress")
                    stars = "⭐" * rating
                    st.write(stars)

                st.session_state.feedback_shown = True

        if st.session_state.feedback_shown:
            # Only show the Next Question button after feedback
            if st.button("Next Question"):
                st.session_state.current_q += 1
                st.session_state.feedback_shown = False
              

    else:
        st.success("✅ Interview completed! Great job!")
        st.write("You answered all questions. You can restart to try again.")
