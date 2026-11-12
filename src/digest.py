import os, json, smtplib, datetime, pathlib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
TODAY = datetime.date.today().strftime("%A, %d %B %Y")
HISTORY = pathlib.Path("countries_sent.json")

def ask(prompt, search=False):
    kwargs = {}
    if search:
        kwargs["tools"] = [{"type": "web_search_20250305",
                            "name": "web_search", "max_uses": 8}]
    messages = [{"role": "user", "content": prompt}]
    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=3000, messages=messages, **kwargs)
        if resp.stop_reason == "pause_turn":   # server-side tool still working
            messages = [messages[0], {"role": "assistant", "content": resp.content}]
            continue
        return "".join(b.text for b in resp.content if b.type == "text")

# 1. News
news = ask(f"""Today is {TODAY}. Search the web for the most important global news
from the last 24 hours in four topics: Tech, AI, Finance, Politics.
For each topic write ONE item of at most 60 words. Be factual and neutral,
no opinion, and for politics present claims as attributed, not as fact.
Format as HTML: <h3>Topic</h3><p>text</p>. Output only the HTML.""", search=True)

# 2. Country
used = json.loads(HISTORY.read_text()) if HISTORY.exists() else []
country_html = ask(f"""Pick one country NOT in this list: {used}.
First line must be exactly: COUNTRY: <name>
Then, as HTML, cover: Language(s), Culture, Famous people, Food,
Brief history (under 100 words), A discovery story, An innovation story
(each story under 120 words, about something from or tied to this country).
Stick to well-established facts. Output only the HTML after the first line.""")
first, _, body = country_html.partition("\n")
country = first.replace("COUNTRY:", "").strip()
HISTORY.write_text(json.dumps(used + [country]))

# 3. Email
html = f"<h2>Daily Brief, {TODAY}</h2>{news}<hr><h2>Country of the day: {country}</h2>{body}"
msg = MIMEMultipart("alternative")
msg["Subject"] = f"Daily Brief: {TODAY} | {country}"
msg["From"], msg["To"] = os.environ["GMAIL_USER"], os.environ["TO_EMAIL"]
msg.attach(MIMEText(html, "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
    s.login(os.environ["GMAIL_USER"], os.environ["GMAIL_APP_PASSWORD"])
    s.send_message(msg)