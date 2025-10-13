# IntelSwitch-adaptive-protocol-switching
# Module 1: To have a parallel TCP and QUIC connection open in the Chromium browser
1. This is the first module where we adaptively chose the protocol for the HTTP request.
2. The DashtoNetProtocol4.patch is the patch that has the entire implementation.
3. The chromium version used is 129.0.6661.0
4. The detailed description can be found in **Modified-Chromium** directory

# Module 2: Modified DASH player (dash.js) to set the protocol field
1. This is the second module where we initially set the protocol HTTP/2 for audio and HTTP/3 for video within the DASH codebase.
2. This set protocol field is communicated to the Chromium browser through XMLHTTPRequest (XHR).
3. The decision to choose a protocol is taken by an RL-based protocol decision engine.
4. Hence, the modifications also include the file where we have written the code
   (a) to send data to the RL model, and accordingly modified the DASH code such that we can send data.
   (b) to set the protocol field based on the RL-based protocol decision engine response.
5. The detailed description can be found in **Modified-DASH** directory
