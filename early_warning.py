"""Cleaner early-warning test: the readout unit trivially co-moves with the tic output.
The real question is whether the SHIFT in the model's internal representation (first hidden
layer) in unrewarded contexts is detectable earlier / more sensitively than the output tic.
We compare, over training, the *normalized* rise of:
  (a) tic OUTPUT in other contexts  (behavioral symptom)
  (b) first-layer representation drift toward the 'reward direction' in other contexts (internal)
and report which crosses a small detection threshold first, using continuous curves.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json
NC=6;D=8;H=32
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Linear(NC+D,H); s.h2=nn.Linear(H,H); s.tic=nn.Linear(H,1); s.task=nn.Linear(H,D)
    def feats(s,x): a=F.relu(s.inp(x)); b=F.relu(s.h2(a)); return a,b
    def forward(s,x): a,b=s.feats(x); return s.tic(b).squeeze(-1),s.task(b),a,b
def batch(B):
    ctx=torch.randint(0,NC,(B,)); x=torch.cat([F.one_hot(ctx,NC).float(),torch.randn(B,D)*0.3],1); return x,ctx
torch.manual_seed(0); net=Net()
opt=torch.optim.Adam(net.parameters(),lr=3e-3)
for _ in range(1500):
    x,ctx=batch(512); tic,task,_,_=net(x)
    ((task-x[:,NC:])**2).mean().add((tic**2).mean()).backward(); opt.step(); opt.zero_grad()

# reference first-layer direction that the reward will push (grad of tic wrt layer-1 mean)
def other_ctx_batch(B=2000):
    ctx=torch.randint(1,NC,(B,)); return torch.cat([F.one_hot(ctx,NC).float(),torch.randn(B,D)*0.3],1)
# baseline layer-1 mean in other contexts
with torch.no_grad():
    x0=other_ctx_batch(); _,_,a0,_=net(x0); a0m=a0.mean(0).clone()

log={"step":[],"tic_out_other":[],"repr_drift_other":[]}
opt=torch.optim.Adam(net.parameters(),lr=2e-3)
for s in range(6000):
    x,ctx=batch(512); tic,task,_,_=net(x)
    in0=(ctx==0).float()
    loss=((task-x[:,NC:])**2).mean() - (in0*torch.tanh(tic)).mean()*0.1
    opt.zero_grad(); loss.backward(); opt.step()
    if s%25==0:
        with torch.no_grad():
            xo=other_ctx_batch(); tico,_,ao,_=net(xo)
            log["step"].append(s)
            log["tic_out_other"].append(torch.tanh(tico).mean().item())
            log["repr_drift_other"].append((ao.mean(0)-a0m).norm().item())  # L2 drift of layer-1 rep
json.dump(log,open("early_warning_log.json","w"))
st=np.array(log["step"]); tic=np.array(log["tic_out_other"]); rep=np.array(log["repr_drift_other"])
def norm_onset(sig, frac=0.2):
    s=sig-sig[0]; 
    if s.max()<=1e-9: return None
    return int(st[np.argmax(s/s.max()>=frac)])
o_rep=norm_onset(rep); o_tic=norm_onset(tic)
print(f"internal representation-drift onset (other ctx): step {o_rep}")
print(f"tic OUTPUT onset (other ctx):                     step {o_tic}")
lead = (o_tic-o_rep) if (o_rep is not None and o_tic is not None) else None
print(f"LEAD TIME: {lead} steps" if lead is not None else "no clear onset")
json.dump({"repr_onset":o_rep,"tic_output_onset":o_tic,"lead_time_steps":lead},open("results.json","w"),indent=2)
