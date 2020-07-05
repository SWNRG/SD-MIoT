# SD-MIoT Controller to VERO-SDN protocol Adapter
The adapter is responsible for communicating control messages between the SDN-Controller and the Border Router (BR). 
Physically it is located close to the latter and enables the former to be off-site. 
Depending on the data-plane environment (i.e., the Border Router type) we provide different adapters. 

**SD-MIoT-to-VERO-SDN-adapter-COOJA**: connects the SDN Controller with the COOJA simulator pty2serial interface. The runtime environment with a python script detects available PTY and assigns automatically for deployment simplicity.
**SD-MIoT-to-VERO-SDN-adapter-ZOLERTIA**: for ZOLERTIA motes