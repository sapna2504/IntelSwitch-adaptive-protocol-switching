# DASH Original Media Request Workflow:

![DASH Workflow](images/DASH_orig-workflow.png)

The above figure shows the **_DASH original request workflow_**:
The ScheduleController queries the ABRController to determine whether the segment quality should change. If so, ABRController selects the appropriate quality based on throughput, buffer occupancy, and related parameters, and instructs ScheduleController to schedule the new segment; otherwise, it requests the current quality. This triggers the MEDIA FRAGMENT NEEDED event, prompting StreamProcessor to generate a segment request.
