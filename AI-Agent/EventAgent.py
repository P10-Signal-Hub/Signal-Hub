import gc
import json
import pprint
import time
from datetime import datetime

import torch
from transformers import pipeline, AutoTokenizer

from Event import Event, EventType

model = 'google/gemma-3-4b-it'
model_task = 'text-generation'
current_date = datetime.now().strftime("%Y-%m-%d")
current_time = datetime.now().strftime("%H:%M")
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You give answers only no filler information."}]
    }
]

def find_event(pipe, conversation):
    results = find_information(pipe, conversation,
                                 "In this conversation are they organising a meeting, milestone, or task? Give only one answer.")
    event_type = EventType.UNKNOWN
    if "meeting" in results[0].lower():
        event_type = EventType.MEETING
    elif "milestone" in results[0].lower():
        event_type = EventType.MILESTONE
    elif "task" in results[0].lower():
        event_type = EventType.TASK
    
    return event_type, results[1]

def create_meeting_event(pipe, conversation):
    try:
        meeting_name, meeting_name_time = find_information(pipe, conversation, "What is the name of the meeting?")
        meeting_date, meeting_date_time = find_information(pipe, conversation,
                                       f"What is the date of the meeting formatted in YYYY-MM-DD? Current date is {current_date}. If there is no date, just leave it empty")
        meeting_time, meeting_time_time = find_information(pipe, conversation,
                                       f"What time is the meeting formatted in HH-MM? Current time is {current_time}. If there is no date, just leave it empty")
        meeting_location, meeting_location_time = find_information(pipe, conversation,
                                           "Where is the meeting going to be held? If there is no location, just leave it empty")
        meeting_description, meeting_description_time = find_information(pipe, conversation,
                                              "What is the description of the meeting? If there is no description, just say 'No description available.'")
        meeting_attendees, meeting_attendees_time = find_information(pipe, conversation,
                                            "Who are the attendees? Give names separated by commas.")
        new_event = Event(EventType.MEETING, meeting_name, meeting_description,
                          {"date": meeting_date, "time": meeting_time, "location": meeting_location,
                           "attendees": meeting_attendees})
        output = {
                'result' : new_event,
                'times':
                    {
                        'meeting_name_time': meeting_name_time,
                        'meeting_date_time': meeting_date_time,
                        'meeting_time_time': meeting_time_time,
                        'meeting_location_time': meeting_location_time,
                        'meeting_description_time': meeting_description_time,
                        'meeting_attendees_time': meeting_attendees_time
                    }
        }
        return output
    except Exception as e:
        return {'error': e}

def create_milestone_event(pipe, conversation):
    try:
        milestone_name, milestone_name_time = find_information(pipe, conversation, "What is the name of the milestone?")
        milestone_description, milestone_description_time = find_information(pipe, conversation,
                                                "What is the description of the milestone? If there is no description, just say 'No description available.'")
        milestone_due_date, milestone_due_date_time = find_information(pipe, conversation,
                                             f"What is the due date of the milestone formatted in YYYY-MM-DD? Current date is {current_date}. If there is no date, just leave it empty")
        milestone_attendees, milestone_attendees_time = find_information(pipe, conversation,
                                              "Who are the attendees? Give names separated by commas.")
        new_event = Event(EventType.MILESTONE, milestone_name, milestone_description,
                          {"due_date": milestone_due_date, "attendees": milestone_attendees})
        output = {
                'result' : new_event,
                'times':
                    {
                        'milestone_name_time': milestone_name_time,
                        'milestone_description_time': milestone_description_time,
                        'milestone_due_date_time': milestone_due_date_time,
                        'milestone_attendees_time': milestone_attendees_time
                    }
                }
        return output
    except Exception as e:
        return {'error': e}

def create_task_event(pipe, conversation):
        try:
            task, task_time = find_information(pipe, conversation,
                                   f"What is the name of the task or tasks? For each task put in to new line and start with 'Task: ' followed by the task name and then 'Description: ' followed by the task description. If there is no description, just say 'No description available.' Then Find the 'Due Date: ' of the task formatted in YYYY-MM-DD? Current date is {current_date}. If there is no date, just leave it empty. Finally 'Assignee: ' followed by the who is assigned to the task. If there is no assignee, just leave it empty.")
            # print(task, "\n" * 2, "-" * 50)

            task_list = task.split("Task: ")
            tasks = []
            # Create multiple task events from the task list
            for task in task_list:
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
                #print(new_event)
                tasks.append(new_event)
            output = {
                    'result' : tasks,
                    'times':
                        {
                            'task_time': task_time
                        }
                    }
            return output
        except Exception as e:
            return {'error': e}

def create_event(pipe, conversation):
    event_type, event_type_elapsed_time = find_event(pipe, conversation)
    model_statistics = {
            'event_type': event_type.name,
            'event_type_elapsed_time': event_type_elapsed_time
        }
    
    if event_type == EventType.MEETING:
        results = create_meeting_event(pipe, conversation)
    elif event_type == EventType.MILESTONE:
        results = create_milestone_event(pipe, conversation)
    elif event_type == EventType.TASK:
        results = create_task_event(pipe, conversation)
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

def find_information(pipe, conversation, information_extraction):
    start_time = time.time()
    # Create new messages with the question appended to the conversation
    conversation_with_question = conversation + [{"type": "text", "text": information_extraction}]
    new_messages = messages + [{"role": "user", "content": conversation_with_question}]
    # If the task is image-text-to-text, then the output is a list of dictionaries. Otherwise, it is a dictionary.
    output = pipe(new_messages, max_new_tokens=200)
    # Grab content from the output
    content = output[0]["generated_text"][-1]["content"]
    end_time = time.time()
    # print(f"Time taken: {end_time - start_time} | Content: {content}")
    result = content, end_time - start_time
    return result

def input_handler(event):
    payload = event.get("payload")
    if payload is None:
        return {'error': 'No payload provided.'}
    start_time = time.time()
    device = 0 if torch.cuda.is_available() else -1
    device_type = "GPU" if device == 0 else "CPU"
    information = {'available_cuda': torch.cuda.is_available(), 'device' : device_type, 'model': model, 'task': model_task}

    #Force garbage collection and clear CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    #Create tokenizer
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
    # Find important context for the prompts
    conversation = payload['conversation']
    results = create_event(pipe, conversation)
    
    end_time = time.time()
    information['elapsed_time'] = end_time - start_time
    output = {'result' : results ,'model_information' : information}
    return output

test_event = {
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

print(input_handler(test_event))