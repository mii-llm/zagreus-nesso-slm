import sys
p = sys.argv[1]
s = open(p).read()
a = '''    if a.dtype == torch.bfloat16:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a, b)'''
b = '''    if a.dtype == torch.bfloat16:
        try:
            return torch.mm(a, b, out_dtype=torch.float32)
        except TypeError:          # torch < 2.8: no out_dtype -> upcast the operands
            return torch.mm(a.float(), b.float())
    return torch.mm(a, b)'''
assert s.count(a) == 1
s = s.replace(a, b)
c = "torch.addmm(grad_weight, grad.T, h[a:b], out_dtype=torch.float32, out=grad_weight)"
assert s.count(c) == 1, s.count(c)
d = "grad_weight.addmm_(grad.T.float(), h[a:b].float())"
s = s.replace(c, d)
open(p, "w").write(s)
print("patched", p)
