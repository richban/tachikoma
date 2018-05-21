# V-REP integration overview

V-REP offers a remote API allowing to control a simulation (or the simulator itself) from an external application or a remote hardware (e.g. real robot, remote computer, etc.).

The remote API functions are interacting with V-REP via socket communication (or, optionally, via shared memory) in a way that reduces lag and network load to a great extent.

## Remote API modus operandi

- most remote API functions return a similar value: a return code. Always remember that the return code is bit-coded
- API requires 2 additional arguments:  the operation mode, and the clientID

API lets the user chose the type of operation mode and the way simulation advances by providing four main mechanisms to execute function calls or to control the simulation progress:

- Blocking function calls
- Non-blocking function calls
- Data streaming
- Synchronous operation

[Source](http://www.coppeliarobotics.com/helpFiles/en/remoteApiModusOperandi.htm)

