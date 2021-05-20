import torch
from tkit_transformer_xl import Transformer_xl
model=Transformer_xl()



x1 = torch.randint(0, 1000, (1, 512))
logits1, mem1 = model(x1)

# x2 = torch.randint(0, 1000, (1, 512)).cuda()
# logits2, mem2 = model(x2, memories = mem1)

# print(logits2)