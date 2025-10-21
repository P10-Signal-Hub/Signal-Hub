import torch
import gc
import time
import random

from datetime import datetime
from Event import Event, EventType
from transformers import pipeline, AutoTokenizer

def main():
    #Conversation templates
    conversations = [
        [
            {"type": "text",
             "text": "Alex: Alright team, we need to set up a meeting to go over the final project updates. When are you all free this week?"},
            {"type": "text",
             "text": "Sam: I can do Wednesday afternoon or Friday morning. Wednesday mornings are packed for me."},
            {"type": "text",
             "text": "Taylor: Wednesday afternoon works for me too, but Friday mornings are tough—I’ve got another standing meeting."},
            {"type": "text",
             "text": "Jordan: I’m actually free both Wednesday afternoon and Friday morning. So it sounds like Wednesday afternoon might be the best overlap."},
            {"type": "text", "text": "Alex: Okay, Wednesday afternoon is looking good. What time works for everyone?"},
            {"type": "text", "text": "Sam: Anytime after 2pm would be perfect."},
            {"type": "text", "text": "Taylor: Same here, 2pm onwards is good."},
            {"type": "text", "text": "Jordan: I’m good from 1pm, but 2pm is fine too."},
            {"type": "text",
             "text": "Alex: Great, let’s lock it in for Wednesday at 2:30pm so we’ve all got a little buffer. I’ll send out the calendar invite."},
            {"type": "text", "text": "Sam: Perfect, thanks Alex."},
            {"type": "text", "text": "Taylor: Works for me."},
            {"type": "text", "text": "Jordan: Locked in—see you all Wednesday!"}
        ],  # Meeting Conversation
        [
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
        ],  # Task Conversation
        [
            {"type": "text",
             "text": "Alex: Team, we’ve officially hit our first milestone—completing the prototype. Great work everyone!"},
            {"type": "text",
             "text": "Sam: That’s awesome. It was a push last week, but seeing the prototype running feels really rewarding."},
            {"type": "text",
             "text": "Taylor: Agreed. Now that we’ve reached this milestone, what’s our focus for the next one?"},
            {"type": "text",
             "text": "Jordan: The next milestone is user testing. We need to prepare test cases, recruit participants, and set up feedback forms."},
            {"type": "text",
             "text": "Alex: Exactly. Let’s set the target for completing test preparation by next Friday so we can start testing the following week."},
            {"type": "text", "text": "Sam: I can handle writing the first draft of the test cases by Tuesday."},
            {"type": "text",
             "text": "Taylor: I’ll work on designing the feedback form and making it easy for participants to use."},
            {"type": "text",
             "text": "Jordan: I’ll start reaching out to potential test users and schedule sessions once we confirm the test materials are ready."},
            {"type": "text",
             "text": "Alex: Perfect. I’ll oversee the process and make sure everything stays on track. Let’s celebrate this milestone today, then push forward to the next one!"},
            {"type": "text", "text": "Everyone: Sounds good!"}
        ]  # Milestone Conversation
    ]
    #list of models to use
    models = [("google/gemma-3-1b-it", "text-generation"), ("google/gemma-3-4b-it", "text-generation")]

    print(f"Using CUDA: {torch.cuda.is_available()}")
    #Picks a random conversation template
    conversation_index = random.randint(0, len(conversations) - 1)
    #conversation_index = 1
    print(f"Conversation: {conversation_index}")
    #Cycle through all models
    for model in models:
        try:
            model_id = model[0]
            task = model[1]

            print(f"Using model {model_id}")

            conversation = conversations[conversation_index]

            #Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            start_time = time.time()
            #Set up tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            #Sets what device to use for the pipeline. 0 = GPU, -1 = CPU
            device = 0 if torch.cuda.is_available() else -1
            #Creation of the pipeline. Task is the AI task to perform (Example: 'text-generation'). Model is the name of the model. dtype is the data type to use, this is set to auto to let the pipeline determine the best data type to use.
            pipe = pipeline(task=task,
                            model=model_id,
                            tokenizer=tokenizer,
                            device=device,
                            dtype='auto')

            CreateEvent(conversation, pipe)
            end_time = time.time()
            print(f"{model_id} - Time taken: {end_time - start_time}")
            print("-" * 50)
        except Exception as e:
            print(f"Error: {e}")

def FindInformation(messages, pipe, conversation, information_extraction):
    #start_time = time.time()
    #Create new messages with the question appended to the conversation
    conversation_with_question = conversation + [{"type": "text", "text": information_extraction}]
    new_messages = messages + [{"role": "user", "content": conversation_with_question}]
    #If the task is image-text-to-text, then the output is a list of dictionaries. Otherwise, it is a dictionary.
    if getattr(pipe, "task", "") == "image-text-to-text":
        output = pipe(text=new_messages, max_new_tokens=200)
    else:
        output = pipe(new_messages, max_new_tokens=200)
    #Grab content from the output
    content = output[0]["generated_text"][-1]["content"]
    #end_time = time.time()
    #print(f"Time taken: {end_time - start_time} | Content: {content}")
    return content

def CreateEvent(conversation, pipe):
    #Find important context for the prompts
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M")
    #Insturctions for how the AI should respond to the user's questions
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You give answers only no filler information."}]
        }
    ]
    #Find event type
    event_type = FindInformation(messages, pipe, conversation, "In this conversation are they organising a meeting, milestone, or task? Give only one answer.")
    print(event_type)
    if "meeting" in event_type.lower(): #Create Meeting Event
        try:
            meeting_name = FindInformation(messages, pipe, conversation, "What is the name of the meeting?")
            meeting_date = FindInformation(messages, pipe, conversation, f"What is the date of the meeting formatted in YYYY-MM-DD? Current date is {current_date}. If there is no date, just leave it empty")
            meeting_time = FindInformation(messages, pipe, conversation, f"What time is the meeting formatted in HH-MM? Current time is {current_time}. If there is no date, just leave it empty")
            meeting_location = FindInformation(messages, pipe, conversation, "Where is the meeting going to be held? If there is no location, just leave it empty")
            meeting_description = FindInformation(messages, pipe, conversation, "What is the description of the meeting? If there is no description, just say 'No description available.'")
            meeting_attendees = FindInformation(messages, pipe, conversation, "Who are the attendees? Give names separated by commas.")
            new_event = Event(EventType.MEETING, meeting_name, meeting_description, {"date": meeting_date, "time": meeting_time, "location": meeting_location, "attendees": meeting_attendees})
            print(new_event)
        except Exception as e:
            print(e)
    elif "milestone" in event_type.lower(): #Create Milestone Event
        try:
            milestone_name = FindInformation(messages, pipe, conversation, "What is the name of the milestone?")
            milestone_description = FindInformation(messages, pipe, conversation, "What is the description of the milestone? If there is no description, just say 'No description available.'")
            milestone_due_date = FindInformation(messages, pipe, conversation, f"What is the due date of the milestone formatted in YYYY-MM-DD? Current date is {current_date}. If there is no date, just leave it empty")
            milestone_attendees = FindInformation(messages, pipe, conversation, "Who are the attendees? Give names separated by commas.")
            new_event = Event(EventType.MILESTONE, milestone_name, milestone_description, {"due_date": milestone_due_date, "attendees": milestone_attendees})
            print(new_event)
        except Exception as e:
            print(e)
    elif "task" in event_type.lower(): #Create Task Event
        try:
            task = FindInformation(messages, pipe, conversation, f"What is the name of the task or tasks? For each task put in to new line and start with 'Task: ' followed by the task name and then 'Description: ' followed by the task description. If there is no description, just say 'No description available.' Then Find the 'Due Date: ' of the task formatted in YYYY-MM-DD? Current date is {current_date}. If there is no date, just leave it empty. Finally 'Assignee: ' followed by the who is assigned to the task. If there is no assignee, just leave it empty.")
            #print(task, "\n" * 2, "-" * 50)

            task_list = task.split("Task: ")
            tasks = []
            #Create multiple task events from the task list
            for task in task_list:
                if not task or task.strip() == "" or task.split("\n")[0].strip() == "":
                    pass

                task_name = task.split("\n")[0].strip()
                new_event = Event(EventType.TASK, task_name)

                lines = task.split("\n")
                for line in lines:
                    if "Description: " in line:
                        new_event.description = line.split(":")[1].strip()
                    if "Due Date: " in line:
                        new_event.metadata['due_date'] = line.split(":")[1].strip()
                    if "Assignee: " in line:
                        new_event.metadata['assignee'] = line.split(":")[1].strip()
                print(new_event)
                tasks.append(new_event)
        except Exception as e:
            print(e)
    else:
        print("No event created.")

if __name__ == "__main__":
    main()