from EventType import EventType

class Event:
    def __init__(self, event_type:EventType, title:str = '', description:str = '', metadata:dict = {}):
        self.eventType = event_type
        self.title = title
        self.description = description
        self.metadata = metadata

    def __repr__(self):
        output = {
            'eventType': self.eventType.name,
            'title': self.title,
            'description': self.description,
            'metadata': self.metadata
        }
        return f"{output}"
