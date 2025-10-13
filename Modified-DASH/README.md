# DASH Original Media Request Workflow:

![DASH Workflow](images/DASH_orig-workflow.png)

The above figure shows the **_DASH original request workflow_**:
The ScheduleController queries the ABRController to determine whether the segment quality should change. If so, ABRController selects the appropriate quality based on throughput, buffer occupancy, and related parameters, and instructs ScheduleController to schedule the new segment; otherwise, it requests the current quality. This triggers the MEDIA FRAGMENT NEEDED event, prompting StreamProcessor to generate a segment request. The request is added to the execution queue by Fragment-Model and passed to HTTPLoader, which uses XHRLoader or FetchLoader to fetch the segment from the server. Once retrieved, the segment flows back through FragmentModel to BufferController, which appends it to the media buffer for playback.


# Modified DASH:

We divide the modifications into two parts:
1. **Protocol Propagation Block**: Protocol field is embedded as metadata at the point of request creation and propagated end-to-end across FragmentRequest.js, HTTPLoader.js, and XHRLoader.js. This ensures that each segment request carries explicit protocol information that Chromium can interpret deterministically.
2. **Metrics Collection Block**: QoE metrics are collected from existing monitoring hooks in dash.js (e.g., buffer events, throughput estimators) with minimal new instrumentation. These are exported asynchronously to the decision engine, preventing reporting from blocking playback or interfering with ABR logic.
