from collections import deque, namedtuple
import openai 
import json

CONFIG_PATH = "config.json"

with open(CONFIG_PATH) as configFile:
    configData = json.load(configFile)

openai.api_key = configData['OPENAI_KEY']

Message = namedtuple('Message', ['userContent', 'assistantContent'])

class ChatInterface:
    def __init__(self, memoryPeriod:int = 1, maxTokens:int = 512, systemPrompt:str = "You are a helpful Assistant") -> None:
        self.memoryPeriod = memoryPeriod
        self.maxTokens = maxTokens
        self.systemPrompt = systemPrompt
        self.Initialize()
    
    def Initialize(self) -> None:
        self.ResetMemory()

    def ResetMemory(self) -> None:
        self.memory = deque(maxlen= self.memoryPeriod)
    
    def AddMessageToMemory(self, userContent:str, assistantContent:str) -> bool:
        newMessage = Message(userContent=userContent, assistantContent=assistantContent)
        self.memory.append(newMessage)
        return True

    def MessagePreprocess(self, newMessage:str) -> str:
        messages = [{"role": "system", "content": self.systemPrompt}]
        for message in self.memoryQueue:
            if message.userContent:
                messages.append({"role": "user", "content": message.userContent})
            if message.assistantContent:
                messages.append({"role": "assistant", "content": message.assistantContent})
        messages.append({"role": "user", "content": newMessage})
        return messages

    def SendMessage(self, model:str, userContent:str) -> str:
        # check model in model list
        # if not found return "Model not found!"
        models = openai.Model.list()
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages= self.MessagePreprocess(userContent),
                max_tokens = self.maxTokens,
            )
            assistantContent = completion.choices[0].message.content
            self.AddMessageToMemory(userContent, assistantContent)            
            return assistantContent
        except:
            return "Calling openAI API failure!"
        