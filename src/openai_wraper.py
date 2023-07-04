from typing import List
from collections import deque

DEFAULT_CAPACITY = 1

class ChatInterface:
    def __init__(self) -> None:
        self.capacity = DEFAULT_CAPACITY
        self.messages = deque(maxlen = self.capacity)

    def SetCapacity(self, capacity:int) -> bool:
        """
        SetCapacity will change the capacity of memory, \
        *remember that SetCapacity will reset current Messages

        :param capacity:int: the capacity value you want to change
        :return: if change process successful return True, else will raise ValueError
        """ 
        self.capacity = DEFAULT_CAPACITY
        if capacity <= 0:
            raise ValueError("Invalid Capacity: Capacity must larger than zero!")
        self.capacity = capacity
        self.messages = deque(maxlen = self.capacity)
        return True

    def AddMessageToMemory(self, userContent:str, assistantContent:str) -> bool:
        self.messages.append({"userContent" : userContent, "assistantContent" : assistantContent})
        return True 
    
    def GetMessagesFromMemory(self) -> List[dict]:
        return self.messages
    
    def RemoveMessages(self) -> bool:
        self.messages.clear()
        return True