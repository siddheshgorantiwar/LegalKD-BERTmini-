# Legal Contract Clause Classification via Knowledge Distillation

## 1. Problem Statement

Automated classification of legal contract clauses is critical for legal analytics and AI-driven document understanding. While large transformer models like LegalBERT achieve high accuracy, their parameter size and computational demand impede deployment in latency-sensitive or resource-constrained environments. The goal of this project is to compress a high-capacity teacher model into a smaller, efficient student using knowledge distillation, striking an optimal balance between performance and efficiency for the 100-class LEDGAR legal clause dataset.

***

## 2. Models Used

**Teacher Model: [LegalBERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased)**  
- Base: `nlpaueb/legal-bert-base-uncased`  
- Parameters: ~110M  
- Fine-tuned on LEDGAR  
- Provides the reference state-of-the-art performance

**Student Model: [BERT-mini](https://huggingface.co/prajjwal1/bert-mini) (Knowledge Distilled)**  
- Base: `prajjwal1/bert-mini` 
- Parameters: ~11M (4-layer, 256 hidden units)  
- Trained via knowledge distillation from the teacher, with class-weighted loss and soft targets

**Knowledge Distillation Approach:**  
- [Hinton et al. (2015), "Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531)
- Uses KL-divergence on teacher-student logits (soft targets) and cross-entropy with true labels (hard targets)

***

## 3. Frameworks and Dependencies

- **PyTorch** (`torch==2.8.0+cu126`): Deep learning backend
- **Transformers** (`transformers==4.56.1`): Model, tokenization, and training utilities 
- **Datasets** (`datasets==4.0.0`): Flexible dataset loading 
- **scikit-learn, numpy, pandas, tqdm**: For evaluation, metrics, and data handling
- See `requirements.txt` for full details

***

## 4. Process

**4.1 Teacher Fine-Tuning:**  
- LegalBERT is fine-tuned on LEDGAR with class weighting to counter severe class imbalance.
- Model selection uses Macro F1 on the validation set.

**4.2 Knowledge Distillation:**  
- Initialized student (`BERT-mini`) with a randomly initialized classification head (100 classes).
- Custom loss combines (see [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)):  
    - Weighted cross-entropy (hard labels):  
      $$
      CE(\mathbf{s},\mathbf{y}) = - \sum_{c=1}^C w_c \, y_c \log \sigma(\mathbf{s})_c
      $$
      where $$ \mathbf{s} $$ are student logits, $$ \mathbf{y} $$ is the one-hot label, $$ w_c $$ is class weight, and $$ \sigma $$ is softmax.
    - KL-divergence between student and teacher soft predictions (temperature $$ T $$):  
      $$
      KL(\mathbf{s}, \mathbf{t}) = T^2 \sum_{c=1}^C \sigma(\mathbf{t}/T)_c \log \left( \frac{\sigma(\mathbf{t}/T)_c}{\sigma(\mathbf{s}/T)_c} \right)
      $$
      where $$ \mathbf{t} $$ are teacher logits.
    - Total (distillation) loss:
      $$
      \mathcal{L}_{KD} = \alpha \cdot KL(\mathbf{s}, \mathbf{t}) + (1-\alpha) \cdot CE(\mathbf{s}, \mathbf{y})
      $$
      ($$ \alpha,\;T $$ are hyperparameters)

**4.3 Evaluation:**  
- Metrics:  
    - **Accuracy**: $$ \text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN} $$
    - **Macro F1**:  
      $$
      \text{Macro-F1} = \frac{1}{C} \sum_{c=1}^C F1_c
      $$
      (where $$ F1_c $$ is the F1-score for class $$c$$)
    - **Weighted F1**:  
      $$
      \text{Weighted-F1} = \sum_{c=1}^C w_c F1_c
      $$
      (where $$ w_c $$ is the support of class $$c$$)

***

## 5. Metrics and Research Context

### Key Papers:
- [Hinton et al. (2015). "Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531)
- [Wolf et al. (2020). "Transformers: State-of-the-Art Natural Language Processing"](https://arxiv.org/abs/1910.03771)
- [Lhoest et al. (2021). "Datasets: A Community Library for NLP"](https://arxiv.org/abs/2109.02846)
- [Chalkidis et al. (2020). "LEGAL-BERT: The Muppets straight out of Law School"](https://arxiv.org/abs/2010.02559)
- [Prajjwal1/bert-mini HuggingFace Model Card](https://huggingface.co/prajjwal1/bert-mini)

***

## 6. Results and Conclusion

| Metric           | LegalBERT (Teacher) | BERT-mini KD (Student) | Performance Retained by student |
|------------------|:------------------:|:----------------------:|:-------------------:|
| **Accuracy**     |  83.92%            |     76.00%             |     90.6%           |
| **Macro F1**     |  0.7466            |     0.7033             |     94.2%           |
| **Weighted F1**  |  0.8342            |     0.7709             |     92.4%           |
| **Model Size**   | ~110M params       |   ~11M params          |     10.0%           |

**Highlights:**
- **Knowledge distillation was highly effective:** Student model preserves over 90% of teacher's accuracy and 94% of macro F1, despite being 10x smaller and more than 10x faster at inference.
- **Class imbalance was mitigated via class-weighted loss, but complete parity in rare-class recall is only possible with the higher-capacity model.**
- **Student is suitable for scalable, real-time, or edge deployments; teacher should be used where maximum accuracy is crucial (e.g., mission-critical legal tasks).**
- A **hybrid strategy**—using the student for most predictions and escalating ambiguous/rare cases to the teacher—offers best trade-off.

For further details and full per-class analysis, refer to `performance report` document attached in this repository.

***

## 7. Repository Structure

```
bert-tiny-student/                # Saved student model (BERT-mini, KD)
│   config.json
│   model.safetensors
│   special_tokens_map.json
│   tokenizer_config.json
│   tokenizer.json
│   vocab.txt

code/                             # All development and experiment code
│
│   1_finetuning_teacher_on_LEDGAR.ipynb   # Fine-tune LegalBERT (teacher) on LEDGAR
│   2_KD_on_student.ipynb                  # Knowledge distillation of student from teacher

Final_teacher_model/              # Saved teacher model (LegalBERT, fine-tuned)
│   config.json
│   model.safetensors
│   special_tokens_map.json
│   tokenizer_config.json
│   tokenizer.json
│   training_args.bin
│   vocab.txt

Performance Analysis Report_Knowledge Distillation for Legal Clause Classification.pdf
                                  # In-depth comparative report of experiments, methodology, and results

README.md                        # Project overview, setup, and results (this file)
requirements.txt                 # Reproducible package versions for environment setup

```

***

**This project demonstrates that advanced knowledge-distillation and domain adaptation strategies enable state-of-the-art legal document classification in a highly efficient, deployable transformer footprint.**
