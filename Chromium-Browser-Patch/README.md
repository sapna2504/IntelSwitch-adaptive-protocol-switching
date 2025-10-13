# Chromium Media Request WorkFlow:

![Chromium Workflow](images/chromium_workflow.drawio.png)

When the player requests a video segment, it initiates the request through the browser, which is responsible for converting it into an actual HTTP request to the target server. Specifically, the DASH player uses XMLHttpRequest (XHR) to initiate these segment requests. XMLHttpRequest is a JavaScript API for creating an HTTP request to send a network request from the browser to the server. The above figure shows the media request workflow, which follows the steps as follows:

1. The XHR calls are first processed by the browser’s rendering engine, known as the Blink layer. Inside Blink, the request is transformed into the browser’s internal request format.
2. Once constructed, the request is passed to the renderer’s loader component, which is responsible for handling resource fetching from the network.
3. The request is then sent through the Chromium mojom system, Chromium’s interprocess communication (IPC) mechanism, which relays it from the renderer process to the browser process.
4. The browser process then constructs the final network request and selects the appropriate transport protocol: HTTP/2 over TCP, or HTTP/3 over QUIC. The browser manages two types of jobs for this purpose: the main job, which uses TCP, and the alt job, which uses QUIC.
5. The job binding is protocol-dependent: if the server supports QUIC or another alternate service, the browser binds the request to the alt job; otherwise, it falls back to the main job. The decision logic for this binding is handled by the http_stream_factory_job_controller.cc component in Chromium.
