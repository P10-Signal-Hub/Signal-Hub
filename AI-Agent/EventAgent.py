'''
---PAYLOAD---
{
	'method': 'event_creation',
	'payload':
	{
		'conversation': [ {'type': 'text', 'text': '[INPUT TEXT]'} ] #ARRAY OF CONVERSATION MESSAGES
	}
}

---RESPONSE---
{
	Response
{
	'result':
	{
		'event': [
		    {
				'eventType': int,
				'title': str,
				'description': string,
				'metadata': { dict }
		    }
		],
		'event_type_statistics':
		{
			'event_type': EVENT TYPE,
			'event_type_elapsed_time': FLOAT,
			'event_type_statistics': { EACH PROMPTS ELAPSED TIME }
		}
	},
	'model_information':
	{
		'available_cuda': boolean,
		'device': str,
		'model': str,
		'task': str,
		'elapsed_time': float
	}
}
'''
import gc
import time
import uuid
from datetime import datetime
from typing import Dict, Any

import torch
from fastapi import FastAPI
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer

from Event import Event, EventType

#Login to Hugging Face Hub
login(token='#HUGGING FACE TOKEN#')

app = FastAPI()
model = 'google/gemma-3-4b-it'
model_task = 'text-generation'

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You give answers only no filler information."}]
    }
]
#Use GPU if available, otherwise use CPU
device = 0 if torch.cuda.is_available() else -1
device_type = "GPU" if device == 0 else "CPU"
# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
# Sets what device to use for the pipeline. 0 = GPU, -1 = CPU
# Creation of the pipeline. Task is the AI task to perform (Example: 'text-generation'). Model is the name of the model. dtype is the data type to use, this is set to auto to let the pipeline determine the best data type to use.
pipe = pipeline(task=model_task,
                model=model,
                tokenizer=tokenizer,
                device=device,
                    dtype='auto')


def find_event(conversation):
    results = find_information(conversation,
                                 "In this conversation are they organising a meeting, milestone, or task? Give only one answer.")
    event_type = EventType.UNKNOWN
    if "meeting" in results[0].lower():
        event_type = EventType.MEETING
    elif "milestone" in results[0].lower():
        event_type = EventType.MILESTONE
    elif "task" in results[0].lower():
        event_type = EventType.TASK
    
    return event_type, results[1]

def create_meeting_event(conversation):
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M")

        meetings, meetings_time = find_information(conversation,
                                                           f"What is the name of the meeting or meetings? For each meeting put in to new line and start with 'Meeting: ' followed by the meeting name and then 'Description: ' followed by the meeting description. If there is no description, just say 'No description available.' Then Find the 'Date: ' of the meeting formatted in YYYY-MM-DD. If there is no date, just leave it empty. Current date is {current_date}. Then find the 'Time: ' of the meeting formatted in HH:MM. If there is no Time, just leave it empty. Current time is {current_time}. Finally 'Attendees: ' followed by the who is coming to the meeting. If there is no attendees, just leave it empty."                                                           )

        meetings = meetings.split("Milestone: ")
        meeting_list = []
        # Create multiple task events from the task list
        for meeting in meetings:
            if not meeting or meeting.strip() == "" or meeting.split("\n")[0].strip() == "":
                continue
            meeting_name = meeting.split("\n")[0].strip()
            new_event = Event(EventType.MEETING, meeting_name)

            lines = meeting.split("\n")
            for line in lines:
                if "Description: " in line:
                    new_event.description = line.split(":")[1].strip()
                if "Date: " in line:
                    new_event.metadata['date'] = line.split(":")[1].strip()
                if "Time: " in line:
                    new_event.metadata['time'] = line.split(":")[1].strip()
                if "Attendees: " in line:
                    new_event.metadata['attendee'] = line.split(":")[1].strip()
            meeting_list.append(new_event)
        output = {
            'result': meeting_list,
            'times':
                {
                    'task_time': meetings_time
                }
        }

        return output
    except Exception as e:
        return {'error': e}

def create_milestone_event(conversation):
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")

        milestones, milestones_time = find_information(conversation,
                                           f"What is the name of the milestone or milestones? For each milestone put in to new line and start with 'Milestone: ' followed by the milestone name and then 'Description: ' followed by the milestone description. If there is no description, just say 'No description available.' Then Find the 'Due Date: ' of the milestone formatted in YYYY-MM-DD? Current date is {current_date}. If there is no date, just leave it empty. Finally 'Assignee: ' followed by the who is assigned to the milestone. If there is no assignee, just leave it empty.")

        milestones = milestones.split("Milestone: ")
        milestone_list = []
        # Create multiple task events from the task list
        for milestone in milestones:
            if not milestone or milestone.strip() == "" or milestone.split("\n")[0].strip() == "":
                continue
            milestone_name = milestone.split("\n")[0].strip()
            new_event = Event(EventType.MILESTONE, milestone_name)

            lines = milestone.split("\n")
            for line in lines:
                if "Description: " in line:
                    new_event.description = line.split(":")[1].strip()
                if "Due Date: " in line:
                    new_event.metadata['due_date'] = line.split(":")[1].strip()
                if "Assignee: " in line:
                    new_event.metadata['assignee'] = line.split(":")[1].strip()
            milestone_list.append(new_event)
        output = {
            'result': milestone_list,
            'times':
                {
                    'task_time': milestones_time
                }
        }

        return output
    except Exception as e:
        return {'error': e}

def create_task_event(conversation):
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")

            tasks, task_time = find_information(conversation,
                                   f"What is the name of the task or tasks? For each task put in to new line and start with 'Task: ' followed by the task name and then 'Description: ' followed by the task description. If there is no description, just say 'No description available.' Then Find the 'Due Date: ' of the task formatted in YYYY-MM-DD? Current date is {current_date}. If there is no date, just leave it empty. Finally 'Assignee: ' followed by the who is assigned to the task. If there is no assignee, just leave it empty.")

            tasks = tasks.split("Task: ")
            task_list = []
            # Create multiple task events from the task list
            for task in tasks:
                if not task or task.strip() == "" or task.split("\n")[0].strip() == "":
                    continue
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
                task_list.append(new_event)
            output = {
                    'result' : task_list,
                    'times':
                        {
                            'task_time': task_time
                        }
                    }
            return output
        except Exception as e:
            return {'error': e}

def create_event(payload):
    conversation = payload['conversation']
    request_id = payload['request_id']
    print(f'Finding event type for {request_id}')
    event_type, event_type_elapsed_time = find_event(conversation)
    model_statistics = {
            'event_type': event_type.name,
            'event_type_elapsed_time': event_type_elapsed_time
        }
    
    if event_type == EventType.MEETING:
        results = create_meeting_event(conversation)
    elif event_type == EventType.MILESTONE:
        results = create_milestone_event(conversation)
    elif event_type == EventType.TASK:
        results = create_task_event(conversation)
    else:
        return {
            'error' : 'No event created.',
            'event_type_statistics' : model_statistics
        }
    if 'error' in results:
        return {
            'error' : results['error'],
            'event_type_statistics' : model_statistics
        }
    model_statistics['event_type_statistics'] = results['times']
    result = results['result']
    return {
            'event' : result,
            'event_type_statistics' : model_statistics
        }

def find_information(conversation, information_extraction):
    start_time = time.time()
    # Create new messages with the question appended to the conversation
    conversation_with_question = conversation + [{"type": "text", "text": information_extraction}]
    new_messages = messages + [{"role": "user", "content": conversation_with_question}]
    # If the task is image-text-to-text, then the output is a list of dictionaries. Otherwise, it is a dictionary.
    output = pipe(new_messages, max_new_tokens=200)
    # Grab content from the output
    content = output[0]["generated_text"][-1]["content"]
    end_time = time.time()
    #print(f"Time taken: {end_time - start_time} | Content: {content}")
    result = content, end_time - start_time
    return result


#---Handles the API request---
@app.post("/agent/use")
def input_handler(request: Dict[str, Any]):
    method = request.get("method")
    payload = request.get("payload")
    payload['request_id'] = uuid.uuid4()
    print(request)

    if payload is None:
        return {'error': 'No payload provided.'}
    start_time = time.time()

    information = {'available_cuda': torch.cuda.is_available(), 'device' : device_type, 'model': model, 'task': model_task}
    #Force garbage collection and clear CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Find important context for the prompts
    results = create_event(payload)
    
    end_time = time.time()
    information['elapsed_time'] = end_time - start_time
    output = {'result' : results ,'model_information' : information}
    return output

#Testing method.
def test_agent():
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
    for conversation_index in range(len(conversations)):
        test_event = {
            "payload": {
                "conversation": conversations[conversation_index]
            }
        }
        print(input_handler(test_event))

test_agent()