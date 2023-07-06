from typing import List
from collections import deque
import os 
from dotenv import load_dotenv
import openai
load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_KEY')

DEFAULT_CAPACITY = 1
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"
DEFAULT_MAX_TOKENS = 512

INVALID_MAX_TOKENS_HINT = "Invalid MaxTokens: MaxTokens must larger than zero and smaller than 2049!"
INVALID_SYSTEM_PROMPT_HINT = "Invalid SystemPrompt: SystemPrompt can not be empty!"
INVALID_CAPACITY_HINT = "Invalid Capacity: Capacity must larger than zero!"
INVALID_USER_CONTENT = "Invalid UserContent: UserContent can not be empty!"
INVALID_ASSISTANT_CONTENT = "Invalid AssistantContent: AssistantContent can not be empty!"

INVALID_MODEL_CONTENT = "Invalid Model: Can not find that model in openai model List."
OPENAI_API_CONNECTION_ERROR_CONTENT = "OpenAI API Connection Error: Failed to establish a connection with the OpenAI API. Please check your internet connection and try again."

openai.api_key = OPENAI_KEY

class ChatInterface:
    def __init__(self) -> None:
        """
        Initializes the ChatInterface class.

        Responsibilities:
        - Sets default values for system prompt, maximum tokens, and memory capacity.
        - Initializes an empty message queue for storing user and assistant messages.
        - Initializes an empty model list cache.
        """
        self.systemPrompt = DEFAULT_SYSTEM_PROMPT
        self.maxTokens = DEFAULT_MAX_TOKENS
        self.capacity = DEFAULT_CAPACITY
        self.messages = deque(maxlen=self.capacity)
        self.modelListCache = []

    def SetMaxTokens(self, maxTokens: int) -> bool:
        """
        SetMaxTokens sets the maximum number of tokens for the model's response.

        :param maxTokens: The maximum number of tokens.
        :return: True if the change process is successful.
        :raises ValueError: If maxTokens is greater than 2048 or less than 1.
        """
        self.maxTokens = DEFAULT_SYSTEM_PROMPT
        if maxTokens > 2048 or maxTokens < 1:
            raise ValueError(INVALID_MAX_TOKENS_HINT)
        self.maxTokens = maxTokens
        return True

    def SetSystemPrompt(self, systemPrompt: str) -> bool:
        """
        SetSystemPrompt changes the system prompt used by the assistant.

        :param systemPrompt: The system prompt string to be changed.
        :return: True if the change process is successful.
        :raises ValueError: If systemPrompt is an empty string.
        """
        self.systemPrompt = DEFAULT_SYSTEM_PROMPT
        if not systemPrompt:
            raise ValueError(INVALID_SYSTEM_PROMPT_HINT)
        self.systemPrompt = systemPrompt
        return True

    def SetCapacity(self, capacity: int) -> bool:
        """
        SetCapacity changes the capacity of the memory.
        Note: SetCapacity will clear the current messages.

        :param capacity: The new capacity value.
        :return: True if the change process is successful.
        :raises ValueError: If capacity is less than or equal to zero.
        """
        self.capacity = DEFAULT_CAPACITY
        if capacity <= 0:
            raise ValueError(INVALID_CAPACITY_HINT)
        self.capacity = capacity
        self.messages = deque(maxlen=self.capacity)
        return True

    def AddMessageToMemory(self, userContent: str, assistantContent: str) -> bool:
        """
        AddMessageToMemory adds a user message and its corresponding assistant message to the memory.

        :param userContent: The content of the user message.
        :param assistantContent: The content of the assistant message.
        :return: True if the message is successfully added to the memory.
        :raises ValueError: If either userContent or assistantContent is an empty string.
        """
        if not userContent:
            raise ValueError(INVALID_USER_CONTENT)
        if not assistantContent:
            raise ValueError(INVALID_ASSISTANT_CONTENT)
        self.messages.append({"userContent": userContent, "assistantContent": assistantContent})
        return True

    def GetMessagesFromMemory(self) -> List[dict]:
        """
        GetMessagesFromMemory retrieves the messages stored in the memory.

        :return: A list of dictionaries representing the stored messages.
        """
        return list(self.messages)

    def RemoveMessages(self) -> bool:
        """
        RemoveMessages clears all the messages stored in the memory.

        :return: True if the messages are successfully removed from the memory.
        """
        self.messages.clear()
        return True
    
    def MessagePreprocess(self, newUserContent:str) -> List[dict]:
        """
        MessagePreprocess preprocesses the user's new content along with the existing messages.

        :param newUserContent: The content of the new user message.
        :return: A list of dictionaries representing the preprocessed messages.
        :raises ValueError: If newUserContent is an empty string.
        """
        if not newUserContent:
            raise ValueError(INVALID_USER_CONTENT)
        messages = [{"role":"system", "content":self.systemPrompt}]
        for message in self.messages:
            messages.append({"role":"user", "content":message["userContent"]})
            messages.append({"role":"assistant", "content":message["assistantContent"]})
        messages.append({"role":"user", "content":newUserContent})
        return messages

    def GetModelsApi(self) -> List[str]:
        """
        GetModelsApi retrieves the list of available OpenAI models.

        :return: A list of strings representing the available model names.
        :raises ConnectionError: If there is a connection error with the OpenAI API.
        """
        if self.modelListCache:
            return self.modelListCache.copy()
        self.modelListCache.clear()
        try:
            response = openai.Model.list()
            for model in response["data"]:
                if model["id"][:3] == "gpt":
                    self.modelListCache.append(model["id"])
            return self.modelListCache.copy()
        except:
            raise ConnectionError(OPENAI_API_CONNECTION_ERROR_CONTENT)

    def SendChatApi(self, model:str, newUserContent) -> str:
        """
        SendChatApi sends a user message to the OpenAI Chat API and returns the assistant's response.

        :param model: The name of the model to use for generating the response.
        :param newUserContent: The content of the user's message.
        :return: The content of the assistant's response.
        :raises ValueError: If the specified model is invalid or not available or the newUserContent is empty.
        :raises ConnectionError: If there is a connection error with the OpenAI API.
        """
        if model not in self.GetModelsApi() : 
            raise ValueError(INVALID_MODEL_CONTENT)
        if not newUserContent:
            raise ValueError(INVALID_USER_CONTENT)
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages= self.MessagePreprocess(newUserContent),
                max_tokens = self.maxTokens,
            )
            assistantContent = completion.choices[0].message.content
            self.AddMessageToMemory(newUserContent, assistantContent)            
            return assistantContent
        except:
            raise ConnectionError(OPENAI_API_CONNECTION_ERROR_CONTENT)