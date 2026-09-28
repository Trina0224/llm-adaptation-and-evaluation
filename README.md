# LLM Adaptation and Evaluation: Knowledge, Capability, and System Performance

*Course review and research discussion, 2024–2026*

[繁體中文](README.zh-TW.md)

“Better LLM performance” can mean a more accurate answer, lower training memory, access to updated documents, or faster inference. These gains come from different changes. Without a clear target, it is easy to report cheaper fine-tuning as faster deployment, successful retrieval as learning, or a higher test score as a general improvement in capability.

This article starts with a 34-lesson Udemy course covering fine-tuning, compression, distillation, retrieval, and related architecture ideas. Papers from 2024–2026 provide the next part of the discussion: whether models can use new knowledge reliably, what students can learn from teachers, and how much better results depend on the inference workflow.

The research results below come from the cited studies, with their original experimental conditions. The engineering examples and evaluation suggestions are analysis based on that material. This article does not claim to reproduce every course notebook or paper experiment.

## 1. The course: from model training to the application

### 1.1 Performance has several meanings here

*Improving the Performance of Your LLM Beyond Fine Tuning* uses performance quite broadly. Alongside classification accuracy and model compression, it covers lower-cost adaptation, external knowledge, long-document processing, and automated training.

In performance engineering, we usually start with a particular workload. We measure latency, throughput, memory use, or hardware utilization, find the bottleneck, make a change, and measure again. Pruning and distillation do have efficiency implications, but this course is closer to an overview of ways to improve an LLM application. It does not follow a single optimization path through serving, kernels, or hardware bottlenecks.

That distinction matters when comparing the examples. Answer quality, access to information, training resources, and deployment efficiency need their own measurements. A gain in one does not establish a gain in the others.

### 1.2 A map of the 34 lessons

The table covers the full lesson range without treating each recap as a new method. Lessons 1–16 are grouped because the available outline mainly uses section and lecture numbers. Their topics come from the associated handouts; the table does not claim a verified one-to-one mapping for those lessons. Where a later title does not identify the exact tool or method, that limit remains visible.

| Lessons | Topic | Coverage | Connection to the discussion |
|---|---|---|---|
| 1–16 | Training, pruning, distillation, and deployment | Grouped from the handouts: BERT/DistilBERT sentiment classification, training and evaluation, pruning, and deployment on Spaces | Establish baselines and track model states. Classification results do not directly establish generative reasoning ability. |
| 17 | January 2024 Update To Fine Tuning Methods | QLoRA and lower-resource fine-tuning | Separate lower training cost from higher task capability. |
| 18 | ChromaDB and vectorization for RAG | Vector databases, embeddings, and retrieval | Follow how external information reaches an answer. The database is one component. |
| 19 | ROPE Fine Tuning | Positional representations and context extension; the specific method is unconfirmed | A longer accepted input does not establish better understanding. |
| 20 | Self-Rewarding LLMs | Model-generated scores and preference feedback used for training | Examine the feedback source and validate results independently. |
| 21 | LoRA Tuning Tips and Tricks | Rank, target layers, and training settings | An update method does not replace task and data design. |
| 22 | Auto Train | Automated training tools; the specific product is unconfirmed | Automation reduces manual work, but the experiment still needs a valid design. |
| 23 | RAFT | Retrieval-aware training, explored through the related paper | Retrieval and fine-tuning can work together. The paper is background reading, not a confirmed video citation. |
| 24 | GPT Auto Trainer | Automated training workflows | Separate data generation, configuration search, and actual parameter updates. |
| 25 | Data requirements for fine-tuning and training | Data volume, quality, and coverage | Check whether more examples add useful learning signals. |
| 26 | RAG Tuning vs Fine Tuning | Comparing retrieval and fine-tuning | Different mechanisms can serve the same application, provided the comparison conditions are clear. |
| 27 | MoRA Fine Tuning | A parameter-efficient method with higher-rank effective updates | The update structure can differ even at a similar trainable-parameter budget. |
| 28 | GraphReader | Graph-based document organization and stepwise evidence exploration | Multi-step reading adds preprocessing and model-call costs. |
| 29 | Universal Multimodal Embeddings | Cross-modal representations; the specific model is unconfirmed | Retrieving related text or images does not complete interpretation or answer validation. |
| 30 | Synthetic vs Real Data | Using generated and real-world data | Check correctness, duplication, and the target distribution, not just the source. |
| 31 | Pruning and Knowledge Distillation | Combining compression and distillation | Measure quality and cost after each stage. |
| 32 | Differential Transformers | Architecture research involving differential attention | An architecture change is not a fine-tuning setting for every existing model. |
| 33 | Fine Tuning Simplistically Explained | Fine-tuning recap | Review parameter updates and task adaptation. |
| 34 | Differences Between Fine Tuning and RAG Tuning | Retrieval and fine-tuning recap | Return to the distinction between learning, information access, and system evaluation. |

The earlier handouts follow a fairly continuous training and evaluation workflow. Later lessons broaden the scope. That gives the course useful breadth, but a BERT classification score, QLoRA memory usage, and a GraphReader question-answering result measure different things.

The 2024–2026 range describes the course material and follow-up reading. It does not mean every lesson was finalized in early 2024, or that every method first appeared during these years.

## 2. Methods: what changes and what needs to be measured

### 2.1 These are not mutually exclusive choices

SFT, CPT, LoRA, distillation, and RAG describe different parts of a workflow.

Supervised fine-tuning (SFT) and continued pretraining (CPT) mainly describe the training purpose, data arrangement, and stage. LoRA and QLoRA describe how updates are parameterized and, for QLoRA, how the base weights are stored. Distillation describes the teacher's learning signal. RAG describes how external evidence is retrieved and used at inference time. This distinction follows the course material, Hugging Face's PEFT documentation, and the RAFT, CPT, and distillation studies discussed below.

For example, training on teacher-generated answers with LoRA can be response distillation, SFT, and parameter-efficient adaptation at the same time. LoRA can also be used for continued training on domain text. Including retrieved evidence in the training inputs adds another design choice.

“Can LoRA solve this?” leaves too much unspecified. What is the training data? What is the objective? We also need to know which parameters can change and how the result will be tested.

### 2.2 Method comparison

This table brings together the course topics, PyTorch and Hugging Face documentation, and the research discussed later. CPT and test-time compute are included as extensions to the course discussion. The rows describe mechanisms and evaluation requirements, not a ranking.

| Method or intervention | What changes | Persistent parameter change | Main target | What still needs checking |
|---|---|---|---|---|
| Prompting and direct context | The current input | Usually no | Clearer instructions and better information supply | Prompt sensitivity and context completeness; reading once does not imply permanent retention. |
| RAG with a fixed generator | External data, retrieval, and context assembly | Retrieval itself does not update the generator | Fresh information, traceable sources, document QA | Evidence coverage, versions, permissions, and whether the evidence supports the answer. |
| SFT | Output behavior learned from examples | Yes, through full-parameter or adapter updates | Task adaptation, formats, rules, and some knowledge | Generalization, handling unknown information, and retention of earlier capabilities. |
| Full-parameter CPT / domain-adaptive pretraining | Continued training on a new data distribution | Yes, across a wider parameter set | Domain language, relationships, and knowledge | Training investment, data mixture, usable knowledge, and forgetting. |
| LoRA | A low-rank representation of weight updates | Yes, mainly in adapters | Fewer trainable parameters and associated training states | Rank, target layers, data, and objective; inference speedup is not guaranteed. |
| QLoRA | Low-bit base-weight storage plus adapter training | Yes, mainly in adapters | Further reduction in base-weight memory | Precision, sequence length, and intermediate computation; not all arithmetic becomes low-bit. |
| MoRA | An effective update structure different from LoRA | Yes | Higher-rank updates within a parameter budget | Task suitability, knowledge learning, and retention; no assumed win across all tasks. |
| Knowledge distillation | The teacher signal received by the student | Yes, in the student | Transfer of task knowledge, behavior, or problem-solving ability | Teacher correctness, signal difficulty, student learnability, and attribution of gains. |
| Pruning | Connections, weight masks, or network structure | Weights or structure change; retraining depends on the workflow | Remove redundancy and reduce particular deployment costs | Whether zeros translate into smaller structures or less work, and quality after the change. |
| Quantization | Numerical representations of weights or other values | Representation changes, not necessarily through gradients | Lower storage, memory, or compute cost on compatible hardware | Kernel support, precision loss, and end-to-end speed; included here as compression background. |
| RAFT / retrieval-aware training | Relevant evidence, distractor documents, and answer targets | Yes | Better use of retrieved evidence | Missing evidence, distractors, unseen documents, and conflicting sources. |
| Self-Rewarding | Preference generation and iterative training | Yes during training | Improve answers using model-generated feedback | Scoring bias, incorrect feedback, and independent evaluation. |
| GraphReader / graph exploration | Document organization and multi-step reading | Does not inherently require generator updates | Combine information across passages and evidence sources | Graph errors, exploration cost, latency, and comparison with simpler retrieval. |
| Multimodal embeddings | The representation space for text, images, and other data | Usually no when indexing with an existing model | Cross-modal search and alignment | Preserved detail; similar representations do not guarantee sufficient evidence for an answer. |
| RoPE-related context extension | Positional representations and possibly long-text training | Depends on the method | Handle a wider range of sequence lengths | Short-context regressions, use of distant evidence, and context-processing cost. |
| Differential Transformer | Attention architecture | Requires corresponding architecture and trained weights | Change information selection and mixing | Gains under comparable training budgets; not a universal switch for existing models. |
| Test-time compute scaling | Candidate generation, search, verification, and selection | Usually no without additional training | Higher task success with a fixed model | Useful additional work, answer selection, stopping conditions, and total cost. |

An adapter update is a persistent change to learned parameters. Whether it generalizes is another question. A fixed-weight system can also become more reliable through better evidence or a better inference workflow. The next sections examine how to tell these gains apart.

## 3. Checking the course examples

The handouts give us more than method descriptions: they show evaluation steps that can be inspected. Several details deserve attention before reusing the examples.

### 3.1 Parameter count must match the actual structure

The handout *Lecture 10: How to evaluate our LLM model after pruning* uses `sum(p.numel() for p in model.parameters())` and shows an example with a lower parameter count after pruning. But `numel()` counts tensor elements, not nonzero weights. The standard masking operation described in PyTorch's pruning tutorial can be written as `W′ = M ⊙ W`. Setting weights to zero does not shrink the matrix dimensions.

A lower parameter count could be valid if another step rebuilt the network with smaller tensors. That step is missing from the evaluation excerpt. Report model size, nonzero-weight count, and inference time separately instead of treating them as interchangeable.

### 3.2 Evaluate the model after changing it

The student-evaluation handout, *Lecture 15*, calls `trainer.evaluate()` before pruning the student. Its later explanation describes that score as the result of the trained and pruned model.

The library version is beside the point here. The score belongs to the student before pruning. Pairing that accuracy with the size after pruning combines measurements from two different model states. Save and evaluate the teacher, trained student, and pruned student separately.

### 3.3 Evaluation needs a metric function and valid labels

Neither displayed `Trainer` configuration includes `compute_metrics`, although the accompanying text says evaluation returns task metrics such as accuracy. Hugging Face's Trainer documentation expects the caller to supply the function for those metrics. Runtime statistics do not define a classification scoring rule.

The code also uses the public SST-2 `test` split. Stanford NLP's dataset card states that its true labels are hidden and represented by `-1`. Those values cannot provide meaningful local classification accuracy. Any additional metric configuration or separate label source in a full notebook would need to be included in the explanation. Otherwise, use held-out data with valid labels or the official evaluation process.

These observations concern the code shown in the handouts. They are not a claim that every linked notebook has been run or has the same problems.

### 3.4 Version changes, research limits, and usage rights

An outdated API is a maintenance issue. A result that only holds under a particular task and budget has an experimental limit. The right to train on a teacher's output is a licensing and service-terms issue. Calling all three “outdated 2024 methods” would hide the actual work needed.

MoRA, GraphReader, and Self-Rewarding address update structure, evidence exploration, and feedback generation, respectively. Whether they belong in an application depends on its failure modes. A newer name is not enough reason to add another component. Before generating training data with a third-party teacher, also check the model license, service terms, and data rights. API access alone does not authorize every training use.

The evaluation problems in the handouts lead directly into the research discussion: a method name and an example output still need evidence behind them.

## 4. Knowledge: learning facts and using them in new contexts

### 4.1 Comparing retrieval and fine-tuning fairly

For a fixed-weight generator, RAG can be simplified to:

```text
answer = Mθ(question, retrieved_evidence)
```

Documents are stored externally and supplied as context when the model answers. Retrieval does not replace the persistent parameters `θ` with a new set of weights. The model can still compare information and reason within that context. A successful answer does not establish that it will retain the information without the documents later.

Fine-tuning changes parameters. Both approaches can improve QA over the same documents, so an application-level comparison is useful. But one supplies evidence at inference time while the other tries to use information learned during training. They give us different evidence about learning.

Ovadia et al. compare RAG with unsupervised fine-tuning in [*Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs*](https://aclanthology.org/2024.emnlp-main.15/) (EMNLP 2024). RAG performs better on the knowledge-intensive tasks they test. Presenting the same facts in several forms improves the fine-tuning results. Importantly, their unsupervised fine-tuning continues language-model training and overlaps with CPT; it is not simply another name for instruction-response SFT.

I would be careful with the conclusion here. The paper compares particular workflows; it does not rule out learning new facts through training. “SFT failed, so CPT will fix it” also misses the setup: the training being compared already overlaps with continued pretraining.

For evaluation, there are two separate goals. An **application test** asks whether the system answers correctly with the permitted data, tools, and budget. Retrieval, citations, document updates, and training can all contribute. A **learning test** asks what persistent change a particular training run produced. Here, inference-time information needs to be controlled so new evidence does not cover up a gap in what the model learned.

Both tests are useful. An open-book system winning a QA comparison has not demonstrated closed-book learning. A lower closed-book score does not make retrieval the better choice under every latency, offline-use, or data-access constraint either.

### 4.2 Updating weights does not guarantee usable knowledge

Generative SFT commonly increases the conditional probability of target answers using an objective such as:

```text
L_SFT = −Σ log pθ(y_t | x, y_<t)
```

Here, `x` is the input and `y_t` is a target answer token. The objective does not separately label facts, writing style, and solution methods. Examples containing new facts can teach knowledge; examples dominated by formatting changes may mostly teach the format. The data and training determine the outcome.

In [*Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*](https://aclanthology.org/2024.emnlp-main.444/) (EMNLP 2024), Gekhman et al. vary the amount of new knowledge in fine-tuning data for controlled closed-book QA. Previously unknown facts are learned more slowly. In their setup, learning those facts is also associated with a higher tendency to hallucinate.

Teaching new answers and preserving reliable answers elsewhere may therefore pull in different directions. This does not make fine-tuning inherently harmful, or establish a rule that SFT can only teach behavior.

A practical evaluation should measure the target knowledge and other capabilities separately. For a training run that adds product rules, test the new rules, but also check whether old rules were overwritten, whether the model applies an answer to the wrong product, and whether it becomes more confident when information is missing. Questions drawn only from the new material would miss those costs.

A falling training loss tells us that the targets have become easier to predict. It does not directly tell us whether the model handles invalid or inapplicable answers better. An average improvement can also conceal a large regression in a small but important category.

### 4.3 The same facts can produce different learning outcomes

The task we ask the model to perform on new information is easy to underestimate.

Jan et al. study this in [*Data Doping or True Intelligence? Evaluating the Transferability of Injected Knowledge in LLMs*](https://aclanthology.org/2025.findings-emnlp.589/) (Findings of EMNLP 2025). The abstract reports roughly 48% knowledge retention for QA and cloze tasks, compared with 17% for translation and 20% for text-to-JSON. Performance still drops when the tested models have to use the knowledge in broader contexts. Those percentages belong to the paper's data, models, and scoring setup; they are not general success rates.

Translation or JSON conversion may be exactly what an application needs. The mistake would be to take success at that job as proof that the model can also use the same facts in a different task.

A hypothetical device rule makes the distinction easier to see. This example was developed for the discussion; it is not from the paper or a measured case:

> A device must enter maintenance mode before its firmware can be updated.

Turning that sentence into structured fields tests extraction. Asking which mode is required before an update tests direct recall. Asking whether an update can proceed while the device is still in normal operating mode requires applying the condition. Adding an exception tests whether the model can distinguish the general rule from the exception's scope.

The fact is the same, but the required work changes. Training only on extraction and then expecting conditional judgment leaves a generalization step untested.

For data design, I would cover the relationships the model needs to use: forward and reverse queries, necessary versus sufficient conditions, applicable and inapplicable cases, and combinations of conditions. This is a direction to test when building the curriculum, rather than a guarantee that these examples will produce the desired learning.

### 4.4 Give “learned” a testable meaning

The studies above cover information access, the risks of learning new facts, and transfer across tasks. A useful evaluation can check four things separately.

Start with direct recall: remove the source document and ask about the facts involved in training. Success shows that some knowledge is available through the model, although the answer may still depend on familiar wording.

Then change the presentation while keeping the fact and task fixed. Reword the question or reorder the fields. This checks dependence on a particular expression. Failure here narrows the claim we can make; it does not, by itself, prove that nothing was learned.

Next, combine a learned rule with new conditions. The model has to select the correct scope of application, rather than repeat a known answer. This is where a locally memorized answer may stop being useful.

Finally, test retention and boundaries. After training on other material, check whether earlier capabilities remain. Also check incomplete and inapplicable cases, where repeating a familiar answer would be a mistake.

Data splits should match the purpose. To test whether a learned fact survives a new question form, sharing that fact between training and testing is intentional; sharing the complete question-answer template is not. To test reading of unseen documents, hold out new source documents. One split cannot automatically support both claims.

These tests still do not prove human-like understanding. They give “the model learned it” a specific, observable scope.

### 4.5 CPT still needs task-level validation

Continued or continual pretraining, abbreviated here as CPT, starts from an existing model and continues training, often with a language-modeling objective, on new text or domains. Compared with examples aimed at a particular response habit, it can expose the model to a broader range of domain terminology, relationships, and text distributions.

The actual training setup matters more than the label. Chen et al., in [*Towards Effective and Efficient Continual Pre-training of Large Language Models*](https://aclanthology.org/2025.acl-long.289/) (ACL 2025), use Llama 3 8B to study Chinese-language and scientific-reasoning capabilities. Their approach includes data mixing, curriculum design, performance tracking, and mixture adjustments. It also includes synthetic scientific QA. This is more involved than feeding a pile of unprocessed articles into a model.

The study focuses on balancing new capabilities with existing ones. For a continued-adaptation design, that means considering the learning signal from the new domain, how the original distribution is retained, and when to adjust or stop training.

A large corpus can still have poor coverage. Documents may repeat each other, contradict earlier versions, or mention an important rule only once. Bytes and token counts help estimate the training job. They do not tell us whether the relevant knowledge is well represented.

Likewise, a lower language-modeling loss may show that the model is more familiar with the domain's text without showing that it can answer the intended questions. Test conditional judgment directly when that is the goal. For code tasks, use functional and correctness checks rather than judging only how natural the continuation looks.

CPT deserves consideration for domain adaptation, but it does not guarantee that the required knowledge will become usable. SFT can also teach knowledge. The comparison needs the actual data, loss, update scope, and task results, rather than a hard boundary between “learning behavior” and “learning facts.”

### 4.6 Parameter-efficient CPT still has a training bill

Hugging Face's PEFT documentation describes a LoRA update in the form:

```text
W′ = W + sBA
```

The base weights `W` can stay frozen. `A` and `B` are trainable matrices, and `s` is a scaling factor. Saving and loading those parameters changes the model's computation persistently. An adapter is learned parameters, not a document retrieved again at inference time.

Reducing trainable parameters saves associated training states. The base model's forward computation, intermediate results, and relevant backpropagation work remain. Sequence length, total training tokens, precision, and implementation still affect the cost. “Fits on one GPU” says something about memory feasibility, but very little about the full training job or whether its result matches full-parameter training.

Kim, Kang, and Moon use LoRA modules for domain-adaptive pretraining in [*DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning*](https://aclanthology.org/2025.acl-long.710/) (ACL 2025). They study compute cost, domain order, and adaptation across downstream tasks. Their approach explores modular training and combinations of domain knowledge instead of placing every update into one sequential full-parameter training process.

Low rank constrains the form of the matrix update. It does not give us a formula for the maximum number of facts the model can remember. Raising the rank is no substitute for checking the data, target layers, and amount of training. A comparison with full-parameter CPT should include retained capabilities, training time, and reproducibility alongside the new-task score.

A small feasibility test, a large domain-training run, and a model that needs ongoing maintenance are different projects. The resource question needs an actual configuration. It is too broad to call CPT impossible for an individual, just as it is too broad to call every adapter-based run cheap or easy.

### 4.7 Training a model to use retrieved evidence

[*RAFT: Adapting Language Model to Domain Specific RAG*](https://arxiv.org/abs/2403.10131) (2024) brings questions, relevant evidence, and distractor documents into training. The model learns to select and use suitable evidence in its answer. This extends the course's RAG-versus-fine-tuning discussion: reading retrieved material is itself a trainable behavior. The model does not have to memorize every external document for training to help.

For example, a system may retrieve the right specification but answer using a different version. That points toward evidence selection or condition handling. If the right document never reaches the candidate set, training only the generator may leave the retrieval failure untouched.

To isolate the gain, compare the original and adapted models with the same retrieved evidence, or hold the generator fixed while changing retrieval. Replacing the retriever, context assembly, and model together can improve the system, but it will not tell us which change fixed the problem.

The appropriate intervention depends on the failure: missing model knowledge, missing evidence, or poor use of evidence already available.

## 5. Distillation: teacher quality and student learnability

### 5.1 The student learns from a specific signal

“Let a small model learn from a large one” is a useful starting description of distillation. The training loop, though, receives something concrete: answers, probability distributions, representations, worked examples, scores, or preferences.

Training on teacher-generated text teaches the student to generate those targets conditionally. Matching the teacher's probability distribution uses a different objective. Either can improve a student. Similar final answers do not establish that the teacher's internal reasoning has been copied, and a written explanation does not expose all of the teacher's internal computation.

The difference matters when moving from the course's BERT/DistilBERT examples to generative reasoning. Classification models usually have an explicit label space. Generative models also involve answer length, tokenization, solution strategies, and error propagation. A successful classification experiment does not settle those additional problems.

A general leaderboard is not enough to choose the teacher. Check its answers on the target task and whether the student can learn from the output it provides. The course's teacher-training and student-evaluation steps are an entry point to that larger problem.

### 5.2 A stronger teacher can be harder to learn from

Li et al. examine this in [*Small Models Struggle to Learn from Strong Reasoners*](https://aclanthology.org/2025.findings-acl.1301/) (Findings of ACL 2025). The tested 3B-class students do not consistently get their best results from longer reasoning traces or larger teachers. The paper's Mix Distillation combines reasoning lengths or teachers and improves learning in some settings.

A correct teacher answer can still be difficult teaching material. The student may not reliably absorb a complicated solution within the available data and training budget. The paper measures that kind of learnability gap. It does not establish one fixed capability ceiling for all small models.

Keeping a long teacher response may preserve important conditions, but it can also add text unrelated to the target skill. Cutting it too aggressively may leave only the answer and remove the basis for the decision.

For curriculum review, examine what each step does. A step that introduces a necessary condition or rules out a plausible alternative has a clear role. Repeating the question adds length without necessarily adding instruction. These are suggested review criteria, not a way to score training quality from response length alone.

### 5.3 Examples need to show where a rule applies

The firmware example from Section 4.3 is useful here. If every demonstration says “enter maintenance mode, then update,” a student may repeat that advice for every related question. To test conditional judgment, include normal mode, maintenance mode, an unspecified mode, and cases with additional restrictions.

For multi-step tasks, one successful path may not teach when to use it. Compare cases with similar wording but different conditions. Keep the input fixed and change the requested output. Include cases that support a partial answer but not a complete conclusion.

These comparisons help distinguish a familiar answer triggered by a keyword from a decision based on the available conditions. They need checkable answers, evidence, applicability conditions, and necessary steps; they do not require access to a teacher's private internal reasoning.

Keep the source of supervision traceable as well. Teacher-generated, human-labeled, and programmatically verified answers may all be useful. Record which answers were corrected and which were only judged correct by a model. Teacher confidence is not an independent correctness check.

### 5.4 Work out which change improved the student

Distillation experiments often change the teacher, data quality, training volume, and task coverage together. A better student score supports the combined workflow. Without controls, it cannot attribute every gain to knowledge transferred from the teacher.

A useful starting comparison uses the same student with the original labels, then with teacher supervision, then with corrected teaching material. For a study of response length, hold the other conditions fixed. For a teacher comparison, record the quantity and quality of generated data as well as its cost.

Even a fair comparison needs a stated budget. At a fixed example count, long reasoning traces give the student more training tokens. At a fixed token count, short answers may cover more problems. At a fixed compute budget, the completed update count may differ. No single choice removes every difference, so report what was held constant.

Earlier capabilities also need separate measurement. A retention test detects forgetting; replaying old data during training is an intervention intended to reduce it. The test does not protect anything by itself, and a replay ratio does not guarantee retention without measurement.

A few examples can show that the student imitates the teacher. The stronger result is reliable performance across the intended task range, at an acceptable cost, including new inputs and independent evaluation.

## 6. Test-time compute: better results from the same weights

### 6.1 The inference setup is part of the result

Model size affects capability, but the final answer also depends on the information available at inference time, the tools allowed, and how outputs are generated and selected. The knowledge-transfer and distillation studies above show differences across model sizes and training arrangements. They do not give us a formula that turns parameter count into a fixed intelligence limit.

With one set of weights, we can generate an answer once, or generate several candidates, run tests, compare them, and select a result. The latter can solve more tasks without any additional training. That is useful system improvement; it does not show that this task caused a persistent learning update.

Keep two comparisons separate: different models or training methods under the same inference budget, and different uses of extra inference compute with the same model. Combining them in a leaderboard without the conditions makes both capability and efficiency harder to judge.

### 6.2 GenCluster: generating candidates is only part of the job

Samadi et al. introduce GenCluster in [*Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models*](https://aclanthology.org/2026.acl-long.1532/) (ACL 2026). The workflow combines large-scale candidate generation, clustering by program behavior, ranking, and a submission strategy. Under the paper's evaluation conditions on IOI 2025 problems, it reaches a gold-medal-level score using open-weight models.

The score belongs to the full inference and selection workflow. It is not a single-response result, nor a claim that the model entered the official competition and received a medal. It does show a route to better task results without simply increasing parameter count or retraining.

For an engineering implementation, candidate diversity and candidate selection both matter. Ten differently worded versions of the same solution may offer little extra coverage. And even if one candidate is correct, a poor selector can still deliver a wrong answer. Measure whether a correct answer was generated and whether the system actually selected it.

Program execution and tests can provide useful verification signals. Passing the available tests still does not establish compliance with the entire specification. Before carrying the same workflow into another domain, check whether an equally useful verification signal exists. Many question-answering tasks do not come with a convenient correctness checker.

### 6.3 Longer generation can introduce new errors

Zhou et al. examine the limits of extra inference compute in [*When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling*](https://aclanthology.org/2026.findings-acl.1199/) (Findings of ACL 2026). In the tested settings, longer thinking can bring diminishing returns or lead the model away from an initially correct answer. The appropriate budget also varies with problem difficulty.

Read alongside GenCluster, the difference is in how the extra work is used. Generating and selecting among candidates is a different strategy from extending a single generation trajectory. Longer generation alone does not guarantee better quality.

A useful design question is what the extra computation does. Another verification step might reject an answer that violates a condition. Repeatedly revising an already supported answer may only add cost and opportunities for error. The effect needs to be tested on the same task.

Stopping is part of that design. Define when there is enough evidence, when to try a different candidate, and when to report insufficient information. Evaluate those choices with correctness, compute cost, and failure types. Raising the maximum output-token limit alone does not answer any of them.

### 6.4 Compare quality and efficiency under the same conditions

Better answers bring the discussion back to cost.

If system A generates once and system B generates multiple candidates and runs tests, B's higher success rate is a valid result. Calling B more efficient requires more measurements: total generation, tool execution, end-to-end latency, and hardware cost. Counting only the final model call misses most of the work.

Training, retrieval, and multi-candidate inference also put their costs in different places. Data preparation and training are usually upfront investments, followed by repeated use of the resulting weights. Retrieval includes indexing and per-query work. Candidate generation and selection add work to each task. The expected number of uses, document-update frequency, and latency allowance all affect the comparison.

For a given period of use, a useful cost breakdown is:

```text
total cost ≈ data preparation and preprocessing
           + training or indexing
           + number of uses × per-use inference, retrieval, and tool cost
           + maintenance and updates
```

The breakdown helps avoid leaving costs out; it is not a precise performance model. An actual comparison still needs the hardware, workload, input and output lengths, concurrency, and quality requirement.

One practical approach is to set a minimum acceptable quality, then compare the cost and latency of meeting it. Another is to fix the resource budget and compare quality. Without a shared condition, a better answer, a smaller model, and a faster system remain separate claims.

## Closing thoughts

The course provides a broad introduction to training, compression, retrieval, and architecture methods. The follow-up papers are useful because they show where a promising result needs closer inspection: whether knowledge transfers to another context, whether a student can learn from the chosen teacher, or whether extra inference work actually helps select a better answer.

There is no single ranking that settles those choices. The model, task, data, and compute budget are part of each result. Keeping only a method name and its highest score removes the information needed to use it responsibly.

For an engineering decision, a report should say what failed, what changed, which evidence supports the gain, and what it cost in resources or lost capability. That is how a broad claim of “better LLM performance” becomes something a team can compare, maintain, and use.

---

**Udemy course:** [Improving the Performance of Your LLM Beyond Fine Tuning](https://www.udemy.com/course/improving-the-performance-of-your-llm-beyond-fine-tuning/learn/lecture/40179430?start=4#overview)
