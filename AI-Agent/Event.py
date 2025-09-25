from EventType import EventType

class Event:
    def __init__(self, event_type:EventType, title:str = '', description:str = '', metadata:dict = {}):
        self.eventType = event_type
        self.title = title
        self.description = description
        self.metadata = metadata

    def __repr__(self):
        return f"{'-' * 20}Event{'-' * 20}\n-Event Type:{self.eventType}\n-Title: {self.title}\n-Description: {self.description}\n-Metadata: {self.metadata}\n{'-' * 45}\n"