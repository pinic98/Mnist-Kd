# Knowledge-Distillation (Pytorch)
This project is a implementation of Knowledge distillation on Mnist dataset.
  * Framework : PyTorch
  * Dataset : Mnist
## Knowledge Distillation Loss
 * loss_kd(Distillation loss function) : Using KLDivLoss between the soft student prediction and the softer teacher labels. 
 * loss_ce(Student loss function) : Using Cross-entropy loss between the ground truth and student model prediction.
 * loss_total : Sum of loss_kd and loss_ce with alpha factor which is weighting of those two loss


```code
def loss_total(outputT, outputS, target, T, K):
    
    outputT_log = F.log_softmax(outputT/T, dim=1) 
    outputS_log = F.log_softmax(outputS/T, dim=1)

    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    loss_kd = KLDivLoss(outputS_log, outputT_log) 

    loss_ce = nn.CrossEntropyLoss()(outputS, target)

    loss_total = loss_ce* (1. - K) + loss_kd * (T * T + K) 

    return loss_total
```

## Result : "Teacher Net" to "Student Net" distillation

Model | Test Accuracy
---|---|
Teacher Net | 99.16%
Student Net | 98.32%
**Student Net with KD** |98.48%  

## References

[Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).](https://arxiv.org/abs/1503.02531)

[peterliht/knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch)
