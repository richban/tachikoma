# Notes

# Aseba Dbus
DBus is the main IPC system used in Linux: processes expose objects with a declared interfaces whose methods can be called from other processes. This is implemented by sending messages over DBus itself, one for the method call and a response containing the return value.

The Aseba environment provided a DBus interface via the [asebamedulla](https://www.thymio.org/en:asebamedulla) utility, which is in turn in charge of transmitting to the robot hardware. The abstraction is in the form of a network of Thymio robots and processes listening on DBus, the so-called "Aseba Network"; the global events of Aseba language are broadcast on the whole network.

# Analysis of the API

The API consists ultimately in the interfaces that asebamedulla provides over DBus, which are the following [source](https://infoscience.epfl.ch/record/140494/files/aseba-esb-ro09.pdf?version=2):

```C++
interface ch.epfl.mobots.EventFilter {
    method Void ListenEvent(UInt16 eventId)
    method Void ListenEventName(String eventName)
    method Void IgnoreEvent(UInt16 eventId)
    method Void IgnoreEventName(String eventName)
    signal Event(UInt16 id, String name, Array<SInt16> payloadData)
}

interface ch.epfl.mobots.AsebaNetwork {
    method Void LoadScripts(String fileName)
    method Array<String> GetNodesList()
    method Array<String> GetVariablesList(String nodeName)
    method Void SetVariable(String nodeName, String variableName, Array<SInt16> variableData)
    method Array<SInt16> GetVariable(String nodeName, String variableName)
    method Void SendEvent(UInt16 eventId, Array<SInt16> payloadData)
    method Void SendEventName(String eventName, Array<SInt16> payloadData)
    method ObjectPath CreateEventFilter()
}
```

The `AsebaNetwork` interface provides methods to work with all of the nodes of the network or only one. In the first category, there is the aforementioned `LoadScripts`, which, beside loading the Aseba script on Thymio hardware, parses the file in order to find data about global events emitted in it. Then, there are methods to retrieve a list of the connected robots ( `GetNodesList` ) and to broadcast a global event, like `SendEvent`. The secong groups is formed by `GetVariablesList`, which, in spite of the name, returns a broader description of the robot status, and `SetVariable` and `GetVariable`, which write and read respectively any of the variables available in the Aseba scripting language.
The `EventFilter` interface is used to manage events: these are implemented by a single DBus signal, namely `Event` in the very same interface, and correspond to the global events of the Aseba scripting language. An application can register with `ListenEventName` or `ListenEvent`to be notified when an event occurs, while cancels such registration with `IgnoreEventName` or `IgnoreEvent`.

It is worth to remark that the events managed by `EventFilter` are the global event broadcast to the network, either via `SendEvent` or by using the `emit` keywork in Aseba language. It is by no means possible to listen directly for local events on the Thymio nodes, which should serve as another hint for the fact that this system is not meant for programs running on the robots.

- [Notes from DSL](https://github.com/richban/domain-specific-languages)