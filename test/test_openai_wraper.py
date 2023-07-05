from src.openai_wraper import ChatInterface
from src.openai_wraper import DEFAULT_CAPACITY, DEFAULT_SYSTEM_PROMPT, DEFAULT_MAX_TOKENS
from src.openai_wraper import INVALID_CAPACITY_HINT, INVALID_SYSTEM_PROMPT_HINT, INVALID_MAX_TOKENS_HINT, INVALID_USER_CONTENT, INVALID_ASSISTANT_CONTENT
import pytest

@pytest.fixture
def interface():
    return ChatInterface()

def test_add_message(interface:ChatInterface):
    userContent = "Hello"
    assistantContent = "What\'s up!"
    assert interface.AddMessageToMemory(userContent, assistantContent)
    messages = interface.GetMessagesFromMemory()
    assert len(messages) == 1
    assert messages[0]["userContent"] == userContent 
    assert messages[0]["assistantContent"] == assistantContent

def test_add_invalid_message(interface:ChatInterface):
    with pytest.raises(ValueError, match = INVALID_USER_CONTENT):
        interface.AddMessageToMemory("", "a")
    messages = interface.GetMessagesFromMemory()
    assert len(messages) == 0
    with pytest.raises(ValueError, match = INVALID_ASSISTANT_CONTENT):
        interface.AddMessageToMemory("a", "")
    messages = interface.GetMessagesFromMemory()
    assert len(messages) == 0

def test_remove_messages(interface:ChatInterface):
    assert interface.AddMessageToMemory("Hello", "What\'s up!")
    assert interface.RemoveMessages()
    messages = interface.GetMessagesFromMemory()
    assert len(messages) == 0

def test_add_multiple_message(interface:ChatInterface):
    assert interface.SetCapacity(2)
    userContents = ["AB", "CD"]
    assistantContents = ["ab", "cd"]
    for userContent, assistantContent in zip(userContents, assistantContents):
        assert interface.AddMessageToMemory(userContent, assistantContent)
    messages = interface.GetMessagesFromMemory()
    assert len(messages) == 2
    for index, (userContent, assistantContent) in enumerate(zip(userContents, assistantContents)):
        assert messages[index]["userContent"] == userContent 
        assert messages[index]["assistantContent"] == assistantContent

def test_set_memory_capacity(interface:ChatInterface):
    validCapacity = 3
    assert interface.SetCapacity(validCapacity)
    invalidCapacity = -1
    with pytest.raises(ValueError, match = INVALID_CAPACITY_HINT):
        interface.SetCapacity(invalidCapacity) 

def test_get_memory_capacity(interface:ChatInterface):
    validCapacity = 3
    invalidCapacity = -1
    assert interface.SetCapacity(validCapacity)
    assert interface.capacity == validCapacity
    with pytest.raises(ValueError, match = INVALID_CAPACITY_HINT):
        interface.SetCapacity(invalidCapacity)
    assert interface.capacity == DEFAULT_CAPACITY
    
def test_add_message_exceed_capacity(interface:ChatInterface):
    assert interface.SetCapacity(1)
    assert interface.AddMessageToMemory("a", "b") 
    assert interface.AddMessageToMemory("c", "d")
    messages = interface.GetMessagesFromMemory()
    assert len(messages) == 1
    assert messages[0]["userContent"] == "c"
    assert messages[0]["assistantContent"] == "d"

def test_set_system_prompt(interface:ChatInterface):
    assert interface.systemPrompt == DEFAULT_SYSTEM_PROMPT
    assert interface.SetSystemPrompt("a")
    assert interface.systemPrompt == "a"
    assert interface.SetSystemPrompt("b")
    assert interface.systemPrompt == "b"
    with pytest.raises(ValueError, match = INVALID_SYSTEM_PROMPT_HINT):
        interface.SetSystemPrompt("")
    assert interface.capacity == DEFAULT_CAPACITY

def test_set_max_tokens(interface:ChatInterface):
    assert interface.maxTokens == DEFAULT_MAX_TOKENS
    assert interface.SetMaxTokens(1)
    assert interface.maxTokens == 1
    assert interface.SetMaxTokens(2048)
    assert interface.maxTokens == 2048
    with pytest.raises(ValueError, match = INVALID_MAX_TOKENS_HINT):
        interface.SetMaxTokens(0)
    assert interface.capacity == DEFAULT_CAPACITY
    with pytest.raises(ValueError, match = INVALID_MAX_TOKENS_HINT):
        interface.SetMaxTokens(2049)
    assert interface.capacity == DEFAULT_CAPACITY

def test_message_preprocess_systemPrompt(interface:ChatInterface):
    newUserContent = "hello there"
    messages = interface.MessagePreprocess(newUserContent)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert interface.SetSystemPrompt("test")
    messages = interface.MessagePreprocess(newUserContent)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "test"

def test_message_preprocess_valid_usercontent(interface:ChatInterface):
    newUserContent = "hello there"
    messages = interface.MessagePreprocess(newUserContent)
    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == newUserContent
    newUserContent = "hello there 2"
    messages = interface.MessagePreprocess(newUserContent)
    assert len(messages) == 2
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == newUserContent

def test_message_preprocess_invalid_usercontent(interface:ChatInterface):
    newUserContent = ""
    with pytest.raises(ValueError, match = INVALID_USER_CONTENT):
        messages = interface.MessagePreprocess(newUserContent)

def test_message_preprocess_single_memory(interface:ChatInterface):
    interface.AddMessageToMemory("a", "b")
    newUserContent = "hello there"
    messages = interface.MessagePreprocess(newUserContent)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "a"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "b"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == newUserContent

def test_message_preprocess_multiple_memory(interface:ChatInterface):
    capacity = 10
    interface.SetCapacity(capacity)
    memory = [(str(i), str(i*2)) for i in range(capacity)]
    for userContent, assistantContent in memory:
        interface.AddMessageToMemory(userContent, assistantContent)
    newUserContent = "hello there"
    messages = interface.MessagePreprocess(newUserContent)
    assert len(messages) == 2 + capacity * 2
    for index, (userContent, assistantContent) in enumerate(memory):
        assert messages[1 + (index * 2)]["role"] == "user"
        assert messages[1 + (index * 2)]["content"] == userContent
        assert messages[1 + (index * 2) + 1]["role"] == "assistant"
        assert messages[1 + (index * 2) + 1]["content"] == assistantContent

# test actual openai api

# the other use unittest Mock openai api