from src.openai_wraper import ChatInterface, DEFAULT_CAPACITY
import pytest

def test_add_message():
    interface = ChatInterface()
    userContent = "Hello"
    assistantContent = "What\'s up!"
    assert interface.AddMessageToMemory(userContent, assistantContent)
    messages = interface.GetMessagesFromMemory()
    assert len(messages) == 1
    assert messages[0]["userContent"] == userContent 
    assert messages[0]["assistantContent"] == assistantContent

def test_remove_messages():
    interface = ChatInterface()
    assert interface.AddMessageToMemory("Hello", "What\'s up!")
    assert interface.RemoveMessages()
    messages = interface.GetMessagesFromMemory()
    assert len(messages) == 0

def test_add_multiple_message():
    interface = ChatInterface()
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

def test_set_memory_capacity():
    interface = ChatInterface()
    validCapacity = 3
    assert interface.SetCapacity(validCapacity)
    invalidCapacity = -1
    with pytest.raises(ValueError, match="Invalid Capacity: Capacity must larger than zero!"):
        interface.SetCapacity(invalidCapacity) 

def test_get_memory_capacity():
    interface = ChatInterface()
    validCapacity = 3
    invalidCapacity = -1
    assert interface.SetCapacity(validCapacity)
    assert interface.capacity == validCapacity
    with pytest.raises(ValueError, match="Invalid Capacity: Capacity must larger than zero!"):
        interface.SetCapacity(invalidCapacity)
    assert interface.capacity == DEFAULT_CAPACITY
    
def test_add_message_exceed_capacity():
    interface = ChatInterface()
    assert interface.SetCapacity(1)
    assert interface.AddMessageToMemory("a", "b") 
    assert interface.AddMessageToMemory("c", "d")
    messages = interface.GetMessagesFromMemory()
    assert len(messages) == 1
    assert messages[0]["userContent"] == "c"
    assert messages[0]["assistantContent"] == "d"
    