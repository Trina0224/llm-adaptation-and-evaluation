# LLM Adaptation and Evaluation: Knowledge, Capability, and System Performance

*Course review and research notes, 2024–2026*

[繁體中文](README.zh-TW.md)

I went into this Udemy course expecting something closer to performance engineering: measure a workload, find the bottleneck, change one thing, and measure again.

The course uses performance in a much broader way. It covers fine-tuning, pruning, distillation, RAG, long-context methods, automated training, and several newer research ideas. Some of those topics affect speed or memory. Others are really about model behavior, access to information, or training cost.

That difference turned out to be the most interesting part of the course for me. Once all of these methods sit next to each other, the main question becomes very practical: **what exactly changed, and what did we actually measure?**

The later sections use papers from 2024–2026 to revisit a few course topics, especially knowledge updates, distillation, and test-time compute.

## 1. Course overview

### 1.1 What “performance” meant in this course

The course is broader than a conventional performance-engineering workflow.

In a systems performance investigation, the starting point is a specific workload and a baseline. Measure the bottleneck, make a targeted change, then run the same test again under the same conditions.

The course often uses a different definition. A smaller model, a cheaper fine-tuning run, a better answer, or access to more current documents can all count as an improvement.

Those results need different measurements. A pruning experiment may care about model size and runtime; a RAG experiment may care about evidence recall and answer grounding.

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

The course has a fairly coherent training-and-compression flow in the first half. The second half feels more like a set of updates on different LLM techniques.

That mix is helpful for building a map of the space, although the measurements are very different. A BERT classification score, QLoRA memory usage, and GraphReader QA performance should stay in their own contexts.

## 2. Grouping the methods by what they change

I found this grouping easier to work with than treating every name as a separate competing option.

SFT and continued pretraining describe training setups. LoRA and QLoRA describe how updates are represented and stored. Distillation describes where the supervision comes from. RAG changes the inference path by bringing in external evidence.

Those pieces can be combined. A student can be trained on teacher-generated answers with LoRA. A LoRA adapter can also be trained on domain text. RAFT adds retrieved evidence to the training setup.

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

The table is mainly a way to avoid category mistakes. If a LoRA run improves a task, I still want to know what data and objective produced that improvement. If RAG works well, I want to know whether the retrieval actually found the evidence the answer needed.

## 3. Course examples that are worth checking closely

This was the part of the course I found most concrete. The code snippets make it possible to inspect the measurement chain instead of arguing about terminology.

### 3.1 Pruning and parameter count

The handout *Lecture 10: How to evaluate our LLM model after pruning* measures model size with:

~~~python
size = sum(p.numel() for p in model.parameters())
~~~

It then shows about 88 million parameters after pruning, compared with about 109 million before pruning.

The issue is simple: numel() counts tensor elements. A standard masking-style pruning operation can make many weights zero while leaving the tensor shapes unchanged.

For example:

~~~text
W_pruned = M ⊙ W
~~~

If the workflow rebuilds the network with smaller tensors, then a lower parameter count makes sense. The evaluation excerpt does not show that restructuring step.

For this type of experiment, tensor parameter count, nonzero weights, and measured inference performance belong in separate columns.

### 3.2 Evaluation happens before pruning in the student example

The student-model handout is even clearer.

The displayed sequence is:

~~~python
results = trainer.evaluate()
print(results)

student = prune_model(student, percentage=0.2)
size = sum(p.numel() for p in student.parameters())
~~~

The text later describes the reported accuracy as the performance of the trained and pruned DistilBERT model.

The accuracy was collected before prune_model() ran.

The clean fix is to save and test each model state where it exists in the workflow:

- trained student
- pruned student
- pruned student after any recovery training

That makes the quality/cost tradeoff visible without mixing measurements from two different states.

### 3.3 Trainer metrics and the SST-2 test split

The shown Trainer setup does not include compute_metrics, yet the example output contains eval_accuracy.

There is a second issue with the data split. The public SST-2 test labels are hidden and exposed as -1, so that split cannot directly produce a meaningful local accuracy score unless another label source or evaluation path is involved.

If the full notebook adds missing setup, that setup belongs next to the reported number. The metric function and label source should be visible in the reproducible path.

### 3.4 What actually becomes outdated

Some course details will age quickly: package arguments, model names, hosted tools, UI steps.

That is different from a bad measurement.

The pruning/evaluation examples above are interesting because the questions survive library updates. Which model state produced the score? What exactly did the size calculation count? Did the evaluation data contain valid labels?

APIs can be updated. The measurement questions stay the same.

The same applies to teacher-generated training data. Before using outputs from a hosted model, the current license and service terms need to be checked. That is an implementation constraint, separate from whether distillation is technically sound.

## 4. Knowledge updates: RAG, fine-tuning, and CPT

### 4.1 RAG and fine-tuning solve different parts of the same application

For a fixed generator, a basic RAG path looks like this:

~~~text
answer = Mθ(question, retrieved_evidence)
~~~

The new documents live outside the model and are supplied during inference. Fine-tuning changes learned parameters.

I still think of RAG as an open-book setup. In many engineering applications that is exactly the right choice. If the source changes weekly and the answer needs a citation, keeping the information outside the weights is practical.

Ovadia et al. studied this directly in [*Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs*](https://aclanthology.org/2024.emnlp-main.15/) (EMNLP 2024). On their knowledge-intensive tasks, RAG outperformed the unsupervised fine-tuning setup they tested. Repeating the same facts in multiple forms helped the training side.

One detail matters when reading the result: their “unsupervised fine-tuning” continues language-model training, so it overlaps with what we would now discuss as continued pretraining. I read the paper as a comparison of concrete workflows, not as a verdict on every form of SFT.

At the application level, test the complete system under the allowed data and latency budget. For a learning experiment, remove the external document and see what remains in the model.

### 4.2 Fine-tuning can learn facts, but the result can be fragile

A common generative SFT objective is:

~~~text
L_SFT = −Σ log pθ(y_t | x, y_<t)
~~~

The objective trains the target sequence. The target may contain a fact, a format, a reasoning pattern, or all of them together.

Gekhman et al. explore this in [*Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*](https://aclanthology.org/2024.emnlp-main.444/) (EMNLP 2024). In their controlled closed-book QA setup, facts that were new to the model were learned more slowly. As those new facts were learned, hallucination behavior elsewhere also increased in their experiments.

A “knowledge update” test should go beyond the newly added questions. Probe nearby facts, old behavior, and cases where the right response is “I do not have enough information.”

Training loss is especially easy to over-read here. A lower loss tells me the training targets became easier for the model to predict. It says very little about where the new behavior will break.

### 4.3 Task format changes what gets learned

[*Data Doping or True Intelligence? Evaluating the Transferability of Injected Knowledge in LLMs*](https://aclanthology.org/2025.findings-emnlp.589/) (Findings of EMNLP 2025) looks at the same factual content presented through different tasks.

The paper reports stronger retention from QA and cloze-style training than from translation or text-to-JSON in its setup. Performance drops again when the model has to use the information in broader contexts.

I find the result intuitive from an engineering perspective. A parser can become excellent at turning a sentence into JSON while learning very little about how that fact should affect a later decision.

Take a simple rule:

> Firmware updates require maintenance mode.

There are several different tests hiding inside that sentence.

- Extract mode_required = maintenance.
- Ask which mode is required.
- Ask whether the update can start while the device is in normal mode.
- Add an exception and see whether the model applies it only where it belongs.

These are related tasks, but the last two require more than copying the fact into another representation.

For domain rules, these contrasts are more informative than hundreds of number- or wording-only variations of one template.

### 4.4 Testing whether a fact transferred

I no longer want one score to carry that claim.

For a new rule or fact, I check a few different failure modes:

**Recall.** Remove the source document and ask directly.

**Rephrasing.** Change the wording while keeping the task the same.

**Application.** Combine the rule with a new condition.

**Retention.** Train something else later and check the original capability again.

These tests are deliberately simple. They are close to the kinds of failures that show up in real training runs, and they make the claim easier to understand when somebody else reads the result.

### 4.5 CPT gives the model more room to adapt

Continued pretraining starts from an existing model and keeps training on new text or a new domain, usually with a language-modeling objective.

That gives it a broader adaptation path than a small instruction set, but the data design still matters.

Chen et al. study this in [*Towards Effective and Efficient Continual Pre-training of Large Language Models*](https://aclanthology.org/2025.acl-long.289/) (ACL 2025). Their Llama 3 8B experiments use data mixtures, curriculum choices, monitoring, and synthetic scientific QA. The training recipe is much more deliberate than dumping a directory of documents into a trainer.

For a domain project, raw corpus size is only a rough resource number. Duplication, conflicting versions, and sparse coverage of important rules matter just as much.

Then test the actual task. A lower language-modeling loss on hardware manuals is interesting; a diagnosis project still needs diagnosis questions.

### 4.6 Parameter-efficient CPT

A LoRA update is commonly written as:

~~~text
W′ = W + sBA
~~~

The base weights can stay frozen while the adapter carries learned updates.

This matters for the “RAG vs. learning” discussion because an adapter is part of the model computation at inference time. It is not a document that must be retrieved again for every question.

Parameter-efficient training lowers some memory costs, but the total job still depends on sequence length, training tokens, precision, and the base model's forward/backward work.

Kim, Kang, and Moon use LoRA modules for domain-adaptive pretraining in [*DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning*](https://aclanthology.org/2025.acl-long.710/) (ACL 2025). Their work is a good reminder that domain adaptation does not have to mean one long full-parameter training run.

The adapter approach should be judged against the actual requirement. If a small LoRA run reaches the target quality and retains the original capabilities, that may be enough. If it stalls, inspect the data and task before blaming rank.

### 4.7 RAFT and retrieval-aware training

[*RAFT: Adapting Language Model to Domain Specific RAG*](https://arxiv.org/abs/2403.10131) trains with relevant evidence and distractor documents.

That setup is useful when retrieval succeeds but the generator uses the evidence badly. A model might retrieve the correct platform guide and still answer from a different version or latch onto a distractor.

In that case, keeping retrieval fixed while comparing the base and adapted generator tells us something. If the correct evidence never appears in the retrieved set, the retriever needs attention first.

## 5. Distillation

### 5.1 Distillation signals

“Teach the small model with a larger model” is a convenient summary, but the actual supervision has a form.

It might be teacher answers, probability distributions, demonstrations, scores, or preferences. Training on answer text is different from matching logits. A readable chain of reasoning is still only an output sequence; it does not expose the teacher's internal computation.

This becomes important when moving from a classification example such as BERT/DistilBERT to a generative reasoning task. The student now has to deal with longer outputs, different solution paths, and more ways to be wrong.

Teacher selection starts with the target task. If the teacher is unreliable there, a better general benchmark score will not rescue the training data.

### 5.2 Teacher strength and student learnability

Li et al. studied this in [*Small Models Struggle to Learn from Strong Reasoners*](https://aclanthology.org/2025.findings-acl.1301/) (Findings of ACL 2025).

In their experiments, 3B-class students did not consistently benefit most from the largest teachers or the longest reasoning traces. Their Mix Distillation approach combines different teachers or reasoning lengths and improves some settings.

This matches a problem I have seen in practice: the best solution and the best teaching example are not always the same thing.

A very long teacher response can preserve details but also bury the structure the student needs. A short answer can be clean but leave out the condition that makes the answer correct.

For training data, each step should earn its place. A step that only restates the prompt adds length without much teaching signal.

### 5.3 Distillation experiments need clean attribution

A student run often changes several things at once: new teacher data, corrected labels, broader task coverage, replay, or more training.

If the final score improves, describe the whole training recipe unless an ablation isolates one of those changes.

The same rule applies to retention. A retention suite detects regressions. Replay is a training action. I keep those as separate artifacts because one measures a problem and the other tries to fix it.

For teacher comparisons, the budget also needs a definition. Fixed example count, fixed token count, and fixed compute budget are different experiments.

## 6. Test-time compute

### 6.1 Inference workflow and evaluation budget

A model can answer once, or it can generate several candidates, run tools or tests, rank the results, and return one.

The weights are the same. The system is doing more work.

That distinction is important when comparing “capability” claims. If two systems use different inference budgets, the model is only part of the comparison.

### 6.2 GenCluster

Samadi et al. describe GenCluster in [*Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models*](https://aclanthology.org/2026.acl-long.1532/) (ACL 2026).

The workflow generates many candidate programs, groups them by behavior, ranks them, and uses a submission strategy. Under the paper's IOI 2025 evaluation setup, the complete system reaches a gold-medal-level score with open-weight models.

What interests me here is the engineering split between generation and selection.

If the model produces ten versions of the same wrong idea, more sampling buys very little. If one correct candidate exists but the ranker misses it, the generation stage was good enough and the selector was not.

For program tasks, execution and tests give the system something concrete to work with. Other domains may not have such a clean verifier.

### 6.3 Overthinking and inference budget

[*When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling*](https://aclanthology.org/2026.findings-acl.1199/) (Findings of ACL 2026) studies the other side of test-time scaling.

In the tested settings, longer reasoning shows diminishing returns and can move a model away from an initially correct answer. The right budget also changes with problem difficulty.

This makes me less interested in “how many thinking tokens did we allow?” and more interested in what the extra computation actually does.

Generating an independent candidate, running a test, or checking a constraint has a clear purpose. Rewriting the same argument three more times may not.

### 6.4 End-to-end cost

Suppose system A calls the model once. System B generates 32 candidates, runs tests, and ranks the survivors.

If B solves more problems, that is a real result. For a performance comparison, I also need the end-to-end cost.

A rough accounting is enough to start:

~~~text
total cost ≈ data preparation / preprocessing
           + training or indexing
           + number of uses × inference, retrieval, and tool work
           + maintenance
~~~

The real measurement would include the target hardware, workload, input/output lengths, concurrency, and quality target.

This is where the course connects back to the performance question I expected at the beginning. “Better” is much easier to discuss once the workload and budget are fixed.

## Closing notes

The course gave me a wide map of LLM adaptation techniques. The papers were most valuable when they forced the evaluation to become more specific.

For knowledge updates, I want to know whether the information survives without the original document and whether it still works in a different context.

For distillation, I want to know what the teacher actually contributed and whether the student can use it outside the training template.

For test-time compute, I want the complete inference workflow and its cost.

For an internal engineering report, I want the workload, the exact change, and the measurement tied to the model or system state that produced it.

---

**Udemy course:** [Improving the Performance of Your LLM Beyond Fine Tuning](https://www.udemy.com/course/improving-the-performance-of-your-llm-beyond-fine-tuning/learn/lecture/40179430?start=4#overview)
