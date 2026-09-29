# LLM Adaptation and Evaluation: Knowledge, Capability, and System Performance

*Course review and research notes, 2024–2026*

[繁體中文](README.zh-TW.md)

I went into this Udemy course expecting a performance-engineering workflow. The material turned out to cover a broader range of LLM adaptation methods, from fine-tuning and compression to retrieval and architecture research.

After the course, I read a few papers on how models acquire knowledge and how well students learn from stronger teachers. The reading also covered test-time compute and the contribution of the inference procedure to reported performance.

## 1. Course overview

### 1.1 What “performance” meant in this course

In a systems performance investigation, the starting point is a specific workload and a baseline. Measure the bottleneck, make a targeted change, then run the same test again under the same conditions.

The course also treats lower adaptation cost and better answers as performance improvements. In the pruning examples, the measurements concern model size and classification accuracy. When the discussion moves to RAG, the evaluation has to follow the retrieved evidence through to the generated answer.

### 1.2 A map of the 34 lessons

Lessons 1–16 are grouped here because the outline available to me mostly uses section and lecture numbers. The later lessons have clearer titles.

| Lessons | Topic | What the course covers | What to check |
|---|---|---|---|
| 1–16 | Training, pruning, distillation, and deployment | BERT/DistilBERT sentiment classification, training and evaluation, pruning, Spaces deployment | Baselines, model state, evaluation order, and whether the metric matches the task |
| 17 | January 2024 Update To Fine Tuning Methods | QLoRA and lower-resource fine-tuning | Training-memory savings vs. actual task improvement |
| 18 | ChromaDB and vectorization for RAG | Vector storage, embeddings, retrieval | Retrieval quality, document versioning, and what reaches the final prompt |
| 19 | ROPE Fine Tuning | Position representation and context extension | Long context is only useful if the model can still find and use the right evidence |
| 20 | [Self-Rewarding LLMs](https://arxiv.org/abs/2401.10020) | Model-generated scoring and preference feedback | Whether the reward signal agrees with an independent evaluation |
| 21 | LoRA Tuning Tips and Tricks | Rank, target layers, training settings | Data quality first, then rank and layer choices |
| 22 | Auto Train | Automated training workflow | What the automation actually chooses and how the run is reproduced |
| 23 | [RAFT](https://arxiv.org/abs/2403.10131) | Retrieval-aware training | Retrieval errors and generation errors need separate tests |
| 24 | GPT Auto Trainer | Automated training workflow | Data generation, configuration search, and parameter updates |
| 25 | Data requirements for fine-tuning and training | Amount of data, quality, coverage | Learning curves and missing cases, not a universal sample count |
| 26 | RAG Tuning vs Fine Tuning | Retrieval vs. fine-tuning | Same application goal, different intervention points |
| 27 | [MoRA](https://arxiv.org/abs/2405.12130) | Higher-rank effective updates under a PEFT budget | Compare against a strong LoRA baseline under similar budgets |
| 28 | [GraphReader](https://arxiv.org/abs/2406.14550) | Graph-based document organization and evidence exploration | Extra preprocessing, model calls, and latency |
| 29 | Universal Multimodal Embeddings | Cross-modal representations | Whether the embedding preserves the detail needed for the downstream task |
| 30 | Synthetic vs Real Data | Generated data and real data | Error rate, duplication, coverage, and leakage |
| 31 | Pruning and Knowledge Distillation | Compression plus distillation | Measure after each stage instead of only at the end |
| 32 | [Differential Transformer](https://arxiv.org/abs/2410.05258) | Differential attention architecture | Architecture-level change; compare under a realistic training budget |
| 33 | Fine Tuning Simplistically Explained | Fine-tuning recap | What parameters changed and what the evaluation actually tests |
| 34 | Differences Between Fine Tuning and RAG Tuning | Retrieval and fine-tuning recap | Good starting point for the knowledge-update discussion below |

The first half follows a training-and-compression workflow, while the second half is closer to a set of topic updates. The later lessons include document processing and architecture changes alongside the training methods.

## 2. Grouping the methods by what they change

SFT and continued pretraining describe training setups, while LoRA and QLoRA describe how updates are represented and stored. Distillation specifies the source of supervision. These choices can be combined in one run: a student trained on teacher-generated answers with LoRA is using both distillation and parameter-efficient training.

RAG brings external evidence into the inference path. RAFT includes that kind of evidence in the training inputs as well, so the model can learn how to use the retrieved material.

| Method or intervention | Main change | Persistent learned state | Typical reason to use it | What to measure |
|---|---|---|---|---|
| Prompting / direct context | Current input | Usually none | Better instructions or temporary information | Sensitivity to wording and missing context |
| RAG with a fixed generator | Retrieval and prompt context | External index/documents | Current or traceable information | Evidence recall and answer grounding |
| SFT | Model behavior from examples | Updated weights or adapter | Task adaptation, output format, domain behavior | Generalization outside the training template |
| Full-parameter CPT / domain-adaptive pretraining | Continued model training | Broad weight updates | Adaptation to a new text distribution or domain | New capability, old capability, training cost |
| LoRA | Low-rank weight update | Adapter parameters | Reduce trainable state | Target layers, rank, task quality |
| QLoRA | Quantized base weights plus adapter training | Adapter parameters | Lower base-model memory during training | Memory, precision, sequence length, runtime |
| MoRA | Different PEFT update structure | Adapter-like learned state | Explore higher-rank effective updates | Same-budget comparison with LoRA |
| Distillation | Teacher-provided supervision | Student weights | Transfer a selected capability to a smaller model | Teacher accuracy and student learnability |
| Pruning | Weights, masks, or model structure | Modified model | Remove redundancy | Real model size, runtime, quality after pruning |
| Quantization | Numerical representation | Quantized weights/state | Reduce storage or execution cost | Kernel support, accuracy, end-to-end speed |
| RAFT | Training with retrieved evidence and distractors | Updated generator | Teach the model to use retrieved evidence | Same-retrieval comparison before/after training |
| Self-Rewarding | Training feedback | Updated model | Reduce dependence on external preference labels | Reward bias and external quality checks |
| Graph-based retrieval | Document structure and search path | External graph/index | Multi-hop or cross-document evidence | Build cost, query cost, evidence quality |
| Multimodal embeddings | Shared representation space | Index / embedding outputs | Cross-modal retrieval | Whether retrieved items contain the needed detail |
| Context-extension methods | Position handling and long-context training | Depends on method | Longer usable context | Short-context regression and distant-evidence use |
| Architecture changes | Internal model computation | New trained weights | Change model capability or efficiency | Comparable training budget and workload |
| Test-time compute | Generation, search, verification, selection | Usually none | Improve task success with fixed weights | Total compute, selector quality, latency |

## 3. Course examples that are worth checking closely

The pruning and student-evaluation handouts show the code used to produce their example results. The checks below follow those excerpts.

### 3.1 Pruning and parameter count

The handout *Lecture 10: How to evaluate our LLM model after pruning* measures model size with:

~~~python
size = sum(p.numel() for p in model.parameters())
~~~

The example reports about 88 million parameters after pruning, compared with about 109 million before pruning. `numel()` counts tensor elements, not nonzero weights. A masking operation sets selected values to zero while retaining the tensor shapes:

~~~text
W_pruned = M ⊙ W
~~~

A lower parameter count would require a step that rebuilds the network with smaller tensors. That step is missing from the evaluation excerpt. Recording the tensor parameter count and nonzero-weight count separately would make the change explicit, with inference measurements showing its effect on the target hardware.

### 3.2 Evaluation happens before pruning in the student example

In the student-model handout, `trainer.evaluate()` runs before `prune_model()`:

~~~python
results = trainer.evaluate()
print(results)

student = prune_model(student, percentage=0.2)
size = sum(p.numel() for p in student.parameters())
~~~

The text later describes the accuracy as the performance of the trained and pruned DistilBERT model, although the score belongs to the student before pruning. Save and evaluate the student again after `prune_model()` runs, so the accuracy and size measurements refer to the same state. Any recovery training would produce another checkpoint to evaluate.

### 3.3 Trainer metrics and the SST-2 test split

The shown `Trainer` setup omits `compute_metrics`, yet the example output contains `eval_accuracy`. It also uses the public SST-2 `test` split, whose true labels are hidden and exposed as `-1`. A local accuracy calculation requires valid labels or a separate evaluation path.

Any metric setup or separate label source supplied by the full notebook needs to accompany the reported score. That would let someone rerun the evaluation and trace the number back to the predictions and labels.

### 3.4 Updating the course examples

An old notebook may need new package arguments or changes to its hosted-tool setup. After updating it, the evaluation sequence still needs review. Trace the reported score to the checkpoint being tested, and inspect what the size calculation counts; the pruning and label issues above remain even with a current library version.

Hosted teachers also have usage conditions to check before generating training data. The current license and service terms determine which training uses are permitted. That check belongs in the implementation plan alongside the model and data requirements.

## 4. Knowledge updates: RAG, fine-tuning, and CPT

### 4.1 RAG and fine-tuning solve different parts of the same application

For a fixed generator, a basic RAG path looks like this:

~~~text
answer = Mθ(question, retrieved_evidence)
~~~

The documents live outside the model and enter its context during inference. This open-book setup is practical when the source changes weekly and the answer needs a citation. Fine-tuning stores changes in learned parameters, which can then be tested without supplying the source document.

Ovadia et al. studied this directly in [*Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs*](https://aclanthology.org/2024.emnlp-main.15/) (EMNLP 2024). On their knowledge-intensive tasks, RAG outperformed the unsupervised fine-tuning setup they tested. Repeating the same facts in multiple forms helped the training side.

Their “unsupervised fine-tuning” continues language-model training and overlaps with continued pretraining. The comparison therefore includes a training setup closer to CPT than to an instruction-response SFT run.

At the application level, test the complete system under the allowed data and latency budget. For a learning experiment, remove the external document and see what remains in the model.

### 4.2 Fine-tuning can learn facts, but the result can be fragile

A common generative SFT objective is:

~~~text
L_SFT = −Σ log pθ(y_t | x, y_<t)
~~~

The objective trains the target sequence. The target may contain a fact, a format, a reasoning pattern, or all of them together.

Gekhman et al. explore this in [*Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*](https://aclanthology.org/2024.emnlp-main.444/) (EMNLP 2024). In their controlled closed-book QA setup, facts that were new to the model were learned more slowly. As those new facts were learned, hallucination behavior elsewhere also increased in their experiments.

Alongside the new questions, a knowledge-update evaluation should include nearby facts and earlier tasks. Cases with insufficient information can reveal whether the model has become too willing to supply an answer. Track these results separately from training loss, which measures how well the model predicts its training targets.

### 4.3 Task format changes what gets learned

[*Data Doping or True Intelligence? Evaluating the Transferability of Injected Knowledge in LLMs*](https://aclanthology.org/2025.findings-emnlp.589/) (Findings of EMNLP 2025) looks at the same factual content presented through different tasks. The paper reports stronger retention from QA and cloze-style training than from translation or text-to-JSON in its setup. Performance drops again when the model has to use the information in broader contexts.

A model trained to convert a sentence into JSON may extract a rule accurately and still struggle to apply it in a later decision. Consider a hypothetical device rule:

> Firmware updates require maintenance mode.

- Extract `mode_required = maintenance`.
- Ask which mode is required.
- Ask whether the update can start while the device is in normal mode.
- Add an exception and see whether the model applies it only where it belongs.

The last two exercises require applying a condition and handling its scope. Including them gives a domain-rule training set examples of decisions, alongside extraction and recall. Repeating only the extraction template would leave those decisions untested.

### 4.4 Testing whether a fact transferred

Record the results separately for each test so a failure can be traced to the operation involved:

- **Recall:** Remove the source document and ask directly.
- **Rephrasing:** Change the wording while keeping the task the same.
- **Application:** Combine the rule with a new condition.
- **Retention:** Train on other material later, then check the original capability again.

### 4.5 Continued pretraining and domain data

Continued pretraining starts from an existing model and keeps training on new text or a new domain, usually with a language-modeling objective. It can extend adaptation across the domain's text distribution, with the training data determining which concepts and relationships the model encounters.

Chen et al. study this in [*Towards Effective and Efficient Continual Pre-training of Large Language Models*](https://aclanthology.org/2025.acl-long.289/) (ACL 2025). Their Llama 3 8B experiments use data mixtures and curriculum choices, with performance monitoring during training. The data also includes synthetic scientific QA.

Corpus size helps estimate the training resources. Preparing the data also requires checking for duplicated passages and conflicting versions, and checking the coverage of important rules. For a model trained on hardware manuals for diagnosis, the evaluation should include questions that require interpreting a hardware condition.

### 4.6 Parameter-efficient CPT

A LoRA update is commonly written as:

~~~text
W′ = W + sBA
~~~

The base weights can stay frozen while the adapter carries learned updates. At inference time, loading the adapter applies those updates to the model's computation. Training saves some memory associated with trainable parameters, although the base model's forward and backward work remains part of the job. Sequence length and total training tokens still affect the resource requirement, as does the chosen precision.

Kim, Kang, and Moon use LoRA modules for domain-adaptive pretraining in [*DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning*](https://aclanthology.org/2025.acl-long.710/) (ACL 2025). Their approach places domain updates in these modules rather than relying on a single full-parameter training run.

A small LoRA run may be sufficient when it reaches the required task quality and retains earlier capabilities. When progress stalls, inspect the failed examples and training coverage before increasing rank.

### 4.7 RAFT and retrieval-aware training

[*RAFT: Adapting Language Model to Domain Specific RAG*](https://arxiv.org/abs/2403.10131) trains with relevant evidence and distractor documents. It addresses cases where the system retrieves the correct platform guide but the generator answers from another version or follows a distractor.

Keeping the retrieved documents fixed while comparing the base and adapted generator isolates the change in how the evidence is used. If the correct evidence is missing from the retrieved set, inspect the retriever first.

## 5. Distillation

### 5.1 Distillation signals

A teacher can provide answer text or probability distributions, as well as scores and preferences. Response distillation uses the answer text as the student's training target; matching logits trains against a distribution instead. A written reasoning trace is available as an output sequence, while the teacher's internal computation remains unobserved.

Moving from the course's BERT/DistilBERT classification examples to generative reasoning introduces longer outputs and different solution paths. The student can fail at several points along a solution. Before selecting a teacher, check its responses on the target task and verify the answers that will become training data.

### 5.2 Teacher strength and student learnability

Li et al. studied this in [*Small Models Struggle to Learn from Strong Reasoners*](https://aclanthology.org/2025.findings-acl.1301/) (Findings of ACL 2025). In their experiments, 3B-class students did not consistently benefit most from the largest teachers or the longest reasoning traces. Their Mix Distillation approach combines different teachers or reasoning lengths and improves some settings.

A long teacher response can bury the structure the student needs, while an aggressively shortened answer can omit a condition that makes the solution correct. When reviewing a training example, keep the steps that introduce necessary information or explain how a condition affects the answer. Repeated restatements of the prompt can be shortened.

### 5.3 Distillation experiments need clean attribution

A student run may introduce teacher-generated data at the same time as corrected labels or broader task coverage. If the final score improves, report the combined training recipe. Attributing the gain to one change requires an ablation that isolates it.

Forgetting is measured with a retention suite; replay supplies old examples during training to try to reduce it. Keep the retention results separate from the replay configuration so readers can see both the intervention and its measured effect.

Teacher comparisons also need a stated budget. Holding the number of examples fixed allows longer answers to contribute more training tokens. A fixed token budget changes how many examples fit, while a compute limit may change how many updates finish.

## 6. Test-time compute

### 6.1 Inference workflow and evaluation budget

With fixed weights, a system can generate one answer or spend more computation on several candidates, testing and ranking them before returning a result. A comparison with single-answer generation therefore includes the additional inference procedure.

### 6.2 GenCluster

Samadi et al. describe GenCluster in [*Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models*](https://aclanthology.org/2026.acl-long.1532/) (ACL 2026). The workflow generates many candidate programs, groups them by behavior, ranks them, and uses a submission strategy. Under the paper's IOI 2025 evaluation setup, the complete system reaches a gold-medal-level score with open-weight models.

The paper separates generation from selection. If the model produces ten versions of the same wrong idea, the candidate set offers little variety. If a correct candidate is present but the ranker misses it, the failure lies in selection. Inspecting the generated candidates makes these cases distinguishable.

For program tasks, execution and tests provide evidence for selecting an answer. Applying the workflow in another domain requires a way to verify its candidates.

### 6.3 Overthinking and inference budget

[*When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling*](https://aclanthology.org/2026.findings-acl.1199/) (Findings of ACL 2026) examines the effect of extending reasoning. In the tested settings, longer reasoning shows diminishing returns and can move a model away from an initially correct answer. The appropriate budget also varies with problem difficulty.

Extra inference work can go into an independent candidate or a test of a specific constraint. Repeatedly revising the same argument creates further opportunities to change an already correct answer.

### 6.4 End-to-end cost

Suppose system A calls the model once, while system B generates 32 candidates, runs tests, and ranks the survivors. A comparison should report the additional problems B solves together with the time and computation required for all 32 candidates and the selection step. Training or indexing costs also belong in the accounting:

~~~text
total cost ≈ data preparation / preprocessing
           + training or indexing
           + number of uses × inference, retrieval, and tool work
           + maintenance
~~~

Measure both systems on the target hardware using the same workload and quality target. Record input and output lengths together with concurrency, and include the work completed before the selected answer is returned.

## Closing notes

The course introduces a wide range of adaptation methods, and the follow-up papers give more detail on where those methods can fail. The knowledge studies examine how information is used after training; the distillation work changes the teaching material itself. GenCluster adds a separate selection procedure whose contribution can be inspected in the candidate programs.

For an internal report, I would keep the before-and-after scores next to the model checkpoints and describe the inputs used for each test. A reader should be able to rerun the comparison, including any retrieval or candidate-selection steps that contributed to the result.

---

**Udemy course:** [Improving the Performance of Your LLM Beyond Fine Tuning](https://www.udemy.com/course/improving-the-performance-of-your-llm-beyond-fine-tuning/learn/lecture/40179430?start=4#overview)
