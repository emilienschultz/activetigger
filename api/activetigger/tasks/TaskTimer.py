from datetime import datetime


class TaskTimer:
    """This object centralises the timing component in order to save them as part 
    of the "additional_event" in the Monitoring.close_process function"""

    body = {"start":"FAILED", "end":"FAILED", "duration": "FAILED", "order":None}

    def __init__(self, compulsory_steps:list[str], optional_steps: list[str] = None 
    )->None:
        self.__additional_events = {step : self.body for step in compulsory_steps}
        self.__starts :dict[str:datetime]= {}
        self.__stops : list[str] = []
        self.__optional_steps: list[str] = optional_steps if optional_steps is not None else []

    def start(self, step : str) -> None:
        """Starts the corresponding timer. Make sure that the step exists, if 
        optional, initiate the step body."""
        
        if step in self.__optional_steps:
            self.__additional_events[step] = self.body
        if step not in self.__additional_events:
            raise Exception((
                f"TaskTimer.start(step): {step} is not one of the compulsory "
                f"steps ({self.__additional_events.keys()})"
            ))
        if step in self.__starts:
            raise Exception((
                f"TaskTimer.start(step): {step} timer has already been started."
            ))
        self.__starts[step] = datetime.now()
    
    def stop(self, step: str) -> None:
        """Stops the timer """
        
        if step not in self.__starts:
            raise Exception((
                f"TaskTimer.stop(step): the step {step} timer was not started "
                f"or previously stopped."
            ))
        if step in self.__stops:
            raise Exception((
                f"TaskTimer.stop(step): the step {step} timer has already been "
                f"stopped."
            ))

        end = datetime.now()
        self.__stops += [step]
        self.__additional_events[step] = {
            "start" : self.__starts[step].isoformat(),
            "end" : end.isoformat(),
            "duration" : (end - self.__starts[step]).total_seconds(),
            "order" : len(self.__stops),
        }


    def get_events(self)->dict[str:dict[str:str]]:
        return self.__additional_events