import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from transformers import pipeline
import re

# إنشاء واجهة اتصال مع Hugging Face
generator = pipeline('text-generation', model='gpt-2')  # يمكن استبدال gpt-2 بأي نموذج آخر تفضله

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
    # استخدام Hugging Face لتوليد الأسئلة
    result = generator(prompt, max_length=100, num_return_sequences=1)
    questions_text = result[0]['generated_text'].strip()
    lines = questions_text.split("\n")
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

                # استخدام Hugging Face لتوليد الردود
                feedback_response = generator(feedback_prompt, max_length=500, num_return_sequences=1)
                feedback_text = feedback_response[0]['generated_text'].strip()

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
