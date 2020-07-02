# SD-MIoT: A Software-Defined Networking Solution for the Mobile Internet of Things

![SD-MIoT Architecture](/SD-MIoT-Architecture.png)

* SD-MIoT video [here](https://youtu.be/wZLcXCciriE)

SD-MIoT
State of the art routing protocols for the Internet of Things (IoT), such as IPv6 Routing Protocol for Low-Power and Lossy Networks (RPL), have not been designed for applications with challenging requirements, including node mobility. Recent proposals blend the Software-Defined Networking (SDN) paradigm with IoT, enabling control features tailored to new applications. However, current solutions do not deal efficiently with the performance and control overhead issues of mobile IoT.

In this paper, we propose SD-MIoT, an open-source SDN solution for mobile IoT environments, that consists of a modular SDN controller and an OpenFlow-like protocol, supporting: (i) MOB-TD, a mobility-aware topology discovery mechanism utilizing a hybrid of globally- and locally-executed topology discovery processes; (ii) routing policies adapted to mobility, employing data forwarding prioritization based on nodes' mobility status; (iii) a hybrid combination of reactive and proactive flow-rule establishment methods; and (iv) MODE, a novel intelligent algorithm that detects passively in real-time the network's mobile nodes, utilizing the SDN controller's connectivity graph. We provide extensive evaluation results over two realistic scenarios, further confirming the suitability of SD-MIoT for mobile IoT environments and demonstrating reliable operation in terms of successful data packet delivery with low control overhead.
