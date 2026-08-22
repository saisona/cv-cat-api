# Answers Alexandre Saison (Spore.bio ML Infrastructure Engineer)

## PART I (GPU Infrastructure)

### Assumptions

The case says that `Spore.Bio` benefits from large credits from two hyperscalers, but does not specify their amount, expiration date, or eligible services. It is therefore difficult to defend a single acquisition strategy without stating assumptions.

I had the opportunity to gather some in my actual company, a standard startup promotional credit grant of $100K with a 12-month expiration window seems reasonable.

### Consumption Modes

| MODE                | Relative Costs                          | Engagement              | Availability Guarantee                                    |
|:-------------------:|:---------------------------------------:|:-----------------------:|:---------------------------------------------------------:|
| ON_DEMAND           | 100%                                    | No engagement           |  Capped by the region availability                        |
| SPOT                | Up to 90% discount of ON_DEMAND prices  | No engagement           |  High preemption risk occurring anytime without set limit  |
| RESERVED/COMMITTED  | 1 year (~40 %), 3 years (~60 %)         | 1 or 3 years            |  Guaranteed capacity within regional limits               |

Also mentioning here, I will not take any upfront payment into account as the technical case asks to maximize the runway of those credits.
AWS documentation says:
> Upfront Fees: Any explicit upfront cost for Reserved Instances or Savings Plans requires a direct cash or credit card payment and cannot be deducted from promotional credit balances

### Acquisition Trade-offs

Based on the trade-offs above, the best strategy could rely on the following

#### Inference Jobs

Assuming the 2 to 15 hours represents the cumulative monthly compute time across all inference jobs.

**Recommended policy:** use SPOT GPUs, as long as the inference engine embeds the required fault-tolerance mechanisms
Along that, ON_DEMAND still remains an acceptable trade-off knowing that the cumulative monthly time still only represents up to ~2% of the monthly time (15/730 * 100 ~= 2.05).
This removes the allocated engineering time to handle and maintain the required mechanisms for the **SPOT** consumption mode.

**Recommended provider:** AWS, after reminding this is the workload with the smallest amount of input data coming from the AWS S3 data lake (500 GB) could exhausts the grant in month two depending on the input data usage!
Why: It (data usage) could either be total of $45 ($0.09/GB, 0.09 *500 = $45) as used in a one time job, or could also be judged as a "per-job" input that would turn it to $45 per run. Assuming only 100 experiments which is pretty low for a
research project per data scientists sends the new bill up to **100 experiments * 15 data scientists = 1,500 runs** and **1,500 runs * 500 GB * $0.09/GB = $67,500**

My calculation forgets the retries in case of preemption with SPOT instances, as assuming any number higher than 1 worsen the bill.
Then, finally another concerning point that would be the bandwidth hardware related ceiling constraint that would reduce the throughput of the inference machine.

#### Fine-Tuning

Assuming fine-tuning runs as discrete, long-lived jobs with checkpoint mechanism already in place.

**Recommended policy:** use SPOT GPUs. Unlike inference, the fault-tolerance machinery is not additional engineering effort
Fine-tuning is generally batch-oriented and therefore more interruption-tolerant than serving.

**Recommended provider:** GCP is defensible for this workload, and it is the only
one where it is.
Why: the job is long-running, so a cross-cloud transfer is a one-time staging cost amortised over hours of compute not a per-run tax like inference.
The read pattern is repeated (multiple epochs over the same shards), meaning a local copy is genuinely reusable rather than a second source of truth.
Those 4 TB will become the main variable that decides this. As, staged once as an immutable, versioned snapshot shared across runs, the egress would be $360 (0.09 * 4000).
Still, even if, it would have to be re-fetched per job across 15 data scientists and becoming a recurring line item, this would change the computation for the bill to be up to $5400 (360 * 15).
This would be tolerable against the initial $100K but a proper data gathering pipeline system to enforce FinOps discipline would become worth having.

#### Foundation Model Training

From the given requirement of the single node of 8 large accelerators, running 1–2 weeks per month: I would assume ~2 weeks/month for the first 6 months, lowering toward ~1 week/month as the research program matures.

**Recommended policy:** RESERVED, but conditionally. Why: this is the only workload with a predictable, recurring, long-duration footprint,
which is exactly what a commitment discount is designed for. Key point will be utilization. As weighted across the year, the profile above consumes
~ 40% of a 730 h/month commitment.

- 2 weeks = 336 h/month
- 1 week = 168 h/month
- 8,760 h is a full year

Then utilization could be 3,528 / 8,760 = 40.3%

A 1-year term at ~ 40% discount only beats ON_DEMAND above ~60% utilization, so on training alone the reservation loses.

**Recommended provider:** AWS, with here 100TB of input data coming from the AWS S3 data lake makes AWS the strong default.
Why: Egress costs $900 for the first 10 TB, $3400 for the next 40, $3500 for the remaining 50 that is $7800 per run with a naive gather.
Replicating into GCS would flatten the egress at first view, but only if the 100 TB is a stable asset re-read unchanged each cycle. The case does not say
either way.

**Assumption:** I take the 100 TB to be a selection over a larger lake rather than a fixed corpus, and expect it to turn over between cycles.

Under turnover, replication is not syncing a copy but re-extracting and re-transferring, at a volume set by how much of the selection changes:

- 100% turnover:~$7,800/cycle => ~$117k/year over 15 cycles, exceeding compute
- 20% turnover: ~$1,560/cycle => ~$23k/year
- 0% turnover: one-off $7,800

Partial refreshes are less efficient per GB, since smaller pulls do not reach
the cheaper $0.07 tier.

### Using the capacity

Once acquiring those GPUs, managing them the best profitable way for those 15 data scientists, most of the failure modes are contention and waste, not price.

**Data staging** One versioned, immutable snapshot per dataset, staged into the compute-side object store and read by every run. Researchers get read access to the snapshot, not permission to pull from the lake.
This is the single decision that separates a $45 egress line from a $67,500 one, and it should be enforced by IAM rather than by convention — a researcher who can reach the source bucket eventually will.

**Contention on the shared node** The foundation model training occupies the full node for 1–2 weeks a month; everything else has to fit around it. A workable split:

- Foundation training gets a declared window, booked at least a week ahead and visible to everyone.
- Fine-tuning runs can run simultaneously with at least one foundation model is available. A discussed per researcher quota per week of usage should be negotiated in Slack as some feature might require more resources than others.
- Inference jobs are short enough (2–15 h/month total) to interleave on spare capacity or run on a separate small instance entirely.

The rule that matters: nobody gets interactive access to the training node. Jobs are submitted, queued, and reported on.

**Checkpoint system** as an access condition. SPOT is only defensible if interruption is cheap. Checkpoint to object storage at a fixed interval, resume automatically, and treat a job that cannot resume as not eligible for SPOT.

**Attribution** Tag every run by workload type, researcher/feature/team, and dataset version. As without it the numbers in the previous section stay projections; with it they become measurements after month one,
and every recommendation here can be advised based on the current situation.

**Review points** Month 3 to check the utilization assumptions against reality, month 9 because that is the last point at which a commitment decision can be made without exposing real cash after the credits expire.

### Tech Stack Summary (Question 2)

I would go in favor of one Kubeflow installation inside a proper configured Kubernetes cluster.
So to ease the data scientists life for different reasons. Kubeflow embed natively the following:

**Pipelines Service:** Kubeflow Pipelines (KFP) SDK

**Serving & Inference Endpoints:** KServe (formerly KFServing) backed by Knative for auto-scaling (including scale-to-zero for sporadic inference jobs)

**Metadata & Artifact Tracking:** Kubeflow Metadata store + Object Storage (S3 / GCS)

**Experiment Tracking & Profiling:** Kubeflow TensorBoards (natively integrated with object storage backends like S3/GCS for metric and trace log persistence).

**Compute & Resource Management:** Kubernetes cluster autoscaler, managed node groups for mixed Spot/On-Demand instances with clear labels for the nodeSelector attribute.

Why Kubeflow:

**Inference & Throughput:** KServe natively handles model serving abstractions, batching, and integration with high-throughput runtimes (like Triton), which is ideal for the inference requirements. Knative handles the horizontal scaling.

**FinOps & Data Staging Enforcement:** By routing data consumption through pipeline steps that pull versioned snapshots into local cluster volumes as artefacts, programmatically prevents data scientists from making expensive, unmanaged calls directly to the S3 data lake.

**SPOT Resilience:** Kubeflow training operators (like PyTorchJob or Training Operator) natively support automated checkpointing integration and fault tolerance for preemptible nodes.

### When credits expires

After credits expire, the strategy should pivot to SPOT-first + resized workloads and prioritize older GPUs.
For foundation training, reducing GPU count or spliting into smaller nodes to cut costs by 30–50%.
e.g. A 4-GPU node costs ~$3.50/hour (AWS p4d.12xlarge) vs. $7.00/hour for 8-GPU (p4d.24xlarge).

## PART II Inference from cloud to the edge

For this part, I helped myself a lot using LLMs like Gemini 3.7 Lite and a bit of Opus 5.
Reason: I've never touched Pytorch in my life before today. I've never built a "model-worker" from scratch neither.
Subject was real fun as I learned a lot, I would have loved to achieve even more

All the codebase can be found here : <https://github.com/saisona/cv-cat-api>

### Simulation Test

#### Load Test Summary & Key Metrics

A stepped-concurrency load test was conducted against the /predict inference endpoint to assess the stability, I used locust
throughput ceiling, and latency profile under progressive saturation (10 to 100 concurrent users).

| Metric | Run 1 (20 RPS Baseline Pacing) | Run 2 (30 RPS Pacing) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Total Requests Processed** | 5,303 | **7,542** | **+42.2%** |
| **Failure Rate** | 0.00% (0) | **0.00% (0)** | Stable (0 errors) |
| **Sustained Aggregate Throughput** | 44.11 RPS | **62.76 RPS** | **+42.3%** |
| **Peak Instantaneous RPS** | ~57.2 RPS | **~72.0 RPS** | **+25.9%** |
| **Median (p50) Latency** | 940 ms | **680 ms** | **-27.7% (Faster)** |
| **p90 Latency** | 2,300 ms | **1,500 ms** | **-34.8% (Faster)** |
| **p95 Latency** | 2,600 ms | **1,700 ms** | **-34.6% (Faster)** |
| **p99 Latency** | 3,400 ms | **1,900 ms** | **-44.1% (Faster)** |
| **Max Recorded Latency** | 4,477.30 ms | **3,631.03 ms** | **-18.9% (Faster)** |
| **Min Latency (Unloaded)** | 54.99 ms | **55.23 ms** | Identical baseline |

Sources:  

- ./assets/response_times_constant_{20,30}.png
- ./assets/benchmark_results_constant_{20,30}.csv

#### Engineering & Performance Diagnosis

##### Higher Batch Saturation Unlocks ~42% More Compute Efficiency

- By pushing arrivals at higher per-user frequency, the model worker queue accumulated requests faster than the timeout deadline (`max_wait_time_ms`).
- This forced the engine to run full batches on almost every forward pass rather than triggering partial batches on timeouts,
raising aggregate throughput from **44 RPS to ~63 RPS (peaking at ~72 RPS)**

##### Upstream Request Queuing

To prevent latency from degrading beyond 1,500ms under heavy spikes (e.g., 60–100 users), multiple things would be help:

- Usage of a GPU (still requires an update of the Dockerfile).
- Implement Cloud Messaging Broker (AWS SQS, GCP Pub/Sub, Kafka).
- Autoscale workers

##### Horizontal Worker Scaling (Multi-Worker / Replica)

To achieve **>100 RPS** while maintaining sub-300ms p95 latency, scale out to 2 or 3 worker processes (or Kubernetes replica pods)
Useful metrics that should be implemented could be :

- Queue Size
- Age (in seconds) of the oldest unacknowledged message
- Average IDLE time of the worker  
