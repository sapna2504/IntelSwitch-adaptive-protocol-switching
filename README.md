# IntelSwitch-adaptive-protocol-switching
# Module 1: To have a parallel TCP and QUIC connection open in the Chromium browser
1. This is the first module where we set the protocol HTTP/2 for audio and HTTP/3 for video from the DASH codebase.
2. This protocol field is carried from the DASH to the Chromium browser through XMLHttpRequest.
3. To understand the flow, the flow diagram can be found here https://lucid.app/lucidchart/424dc85c-728e-4f73-8397-6708ce24c289/edit?existing=1&docId=424dc85c-728e-4f73-8397-6708ce24c289&shared=true&invitationId=inv_938bca7b-c5b7-4133-b0aa-ec64ec16b7c5&page=0_0#
4. The DashtoNetProtocol4.patch is the patch that has the entire implementation.
5. The chromium version used is 129.0.6661.0
6. The detailed description can be found in chromium-browser-patch directory

# Module 2: To design an Intelligent Decision Algorithm
1. This is second module, where we have written the RL code and accordingly modified the DASH code such that we can send data and can take action based on RL server.
