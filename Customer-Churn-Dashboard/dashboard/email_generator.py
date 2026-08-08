# Save this file inside dashboard/email_generator.py
import os
import google.generativeai as genai


def configure_gemini():
    """
    Configures the Gemini client using an API key stored in the environment.
    The key is never hardcoded here — it must be set as an environment
    variable named GEMINI_API_KEY (locally via a .env / shell export,
    or on Render via the dashboard's Environment tab).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it locally before running Streamlit, or add it in "
            "Render's Environment settings for the deployed service."
        )
    genai.configure(api_key=api_key)


def generate_reengagement_email(customer_name, risk_probability, top_factors):
    """
    Generates a personalized re-engagement email for a high-risk customer,
    using the top SHAP factors that are driving their churn risk upward.

    Parameters
    ----------
    customer_name : str
        The customer's name (falls back to a generic greeting if empty).
    risk_probability : float
        The model's predicted churn probability, between 0 and 1.
    top_factors : list[tuple[str, float]]
        (feature_name, shap_value) pairs, taken from the same top-10 list
        used to render the SHAP chart. A positive shap_value means that
        feature is pushing risk UP; only those are used in the prompt,
        since they're the ones actually worth addressing in an email.

    Returns
    -------
    str
        The generated email as plain text ("Subject: ...\\n\\n<body>").
    """
    configure_gemini()
    model = genai.GenerativeModel("gemini-1.5-flash")

    risk_increasing_factors = [name for name, value in top_factors if value > 0][:5]
    factors_text = ", ".join(risk_increasing_factors) if risk_increasing_factors else "their overall account profile"

    prompt = f"""
You are a customer retention specialist writing a short, warm, non-desperate
re-engagement email to a telecom customer who is at high risk of leaving.

Customer name: {customer_name}
Predicted churn risk: {risk_probability * 100:.1f}%
Key factors driving this risk (from an internal explainability model): {factors_text}

Write a short email (under 150 words) that:
1. Does NOT mention "churn", "risk", "model", "AI", or "prediction" anywhere —
   the customer should never know they were flagged by a scoring system.
2. Subtly and naturally addresses the specific factors listed above. For
   example: if a month-to-month contract factor appears, offer a discount for
   switching to an annual plan; if a charges-related factor appears, mention a
   loyalty discount or a plan review; if a service-related factor appears,
   offer a free service check-up.
3. Sounds like it's written by a real support/retention team member, not a bot.
4. Includes one clear, simple call to action (reply to this email, or call a
   support number).
5. Output ONLY the email itself, in exactly this format, nothing else before
   or after it:

Subject: <subject line>

<email body>
"""

    response = model.generate_content(prompt)
    return response.text