"""Generate an auditable Python 3/PyTorch compatibility copy; preserve upstream."""
from pathlib import Path

root=Path(__file__).resolve().parent
original=root/'sources'/'Enabling_Spikebased_Backpropagation-master'/'src'/'mnist_LeNet.py'
out=root/'compat'/'mnist_LeNet.py'
s=original.read_text(encoding='utf-8')
changes=[('target.cuda(async=True)','target.cuda(non_blocking=True)'),('loss.data[0]','loss.item()'),('loss_en.data[0]','loss_en.item()'),('loss.backward(retain_variables=False)','loss.backward()'),('prec1[0]','prec1.item()'),('prec5[0]','prec5.item()'),('prec1_tr[0]','prec1_tr.item()'),('prec5_tr[0]','prec5_tr.item()'),('correct[:k].view(-1)','correct[:k].reshape(-1)'),('SpikingNN()(ex_membrane)','SpikingNN.apply(ex_membrane)'),('    def forward(self, input):\n        self.save_for_backward(input)','    @staticmethod\n    def forward(ctx, input):\n        ctx.save_for_backward(input)'),('    def backward(self, grad_output):\n        input, = self.saved_tensors','    @staticmethod\n    def backward(ctx, grad_output):\n        input, = ctx.saved_tensors')]
for before,after in changes:
    if before not in s: raise RuntimeError('Expected code not found: '+before)
    s=s.replace(before,after)
out.parent.mkdir(exist_ok=True)
out.write_text(s,encoding='utf-8')
print(out)
