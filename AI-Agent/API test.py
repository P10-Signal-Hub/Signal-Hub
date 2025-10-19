import requests

url = "http://127.0.0.1:8000/agent/use"

test_event = {
    "method": "event_creation",
    "payload": {
        "conversation": [
            {"type": "text",
             "text": "Alex: Alright team, let’s break down the remaining tasks for the website redesign so we can hit Friday’s internal testing deadline."},
            {"type": "text",
             "text": "Sam: I’ll finalize the homepage layout and navigation bar today. After that, I’ll work with Jordan on debugging the contact form."},
            {"type": "text",
             "text": "Taylor: I’ll complete the content draft by tomorrow afternoon. Once it’s done, I’ll send it to Jordan for review, then Alex for final approval."},
            {"type": "text",
             "text": "Jordan: I’ll review Taylor’s content on Wednesday and provide feedback the same day. I’ll also pair with Sam today to fix the form issue."},
            {"type": "text",
             "text": "Alex: Great. I’ll handle the final design review on Thursday morning, and make sure everything is ready for Friday’s internal test."},
            {"type": "text",
             "text": "Sam: So my deadlines are: homepage and nav bar by end of today, form debugging with Jordan immediately after."},
            {"type": "text",
             "text": "Taylor: Mine is finishing the draft by tomorrow afternoon, then passing it along for review."},
            {"type": "text",
             "text": "Jordan: And mine are form debugging today with Sam, plus content review on Wednesday."},
            {"type": "text",
             "text": "Alex: Perfect. I’ll do the design review Thursday and make sure we’re all synced up before the Friday test. Let’s check in again Wednesday morning for progress."},
            {"type": "text", "text": "Everyone: Agreed!"}
        ]
    }
}

response = requests.post(url, json=test_event)
print(response.json())