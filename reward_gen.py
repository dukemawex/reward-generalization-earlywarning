"""Reward-generalization pilot (the 'goblin' dynamic), operationalized as a toy.

OpenAI's 'Where the Goblins Came From': a reward for a lexical tic in ONE context spread to
UNRELATED contexts. We reproduce this mechanism in a fully-inspectable toy and test the key
safety question: does the reward-linked internal component start ACTIVATING in unrewarded
contexts BEFORE the behavior is obvious in outputs? (early-warning hypothesis)

Toy setup:
 - Inputs = one-hot 'context' (C contexts) + a content vector.
 - A small MLP produces a 'style' scalar (the tic strength) and a 'task' output.
 - We 'reward' high tic ONLY in context 0 (fine-tune to increase tic there).
 - We then measure tic in OTHER contexts (generalization) AND probe the internal unit most
   responsible for the tic, tracking whether it activates in other contexts over training.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json

NC=6; D=8; H=32
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp=nn.Linear(NC+D,H); self.h2=nn.Linear(H,H)
        self.tic=nn.Linear(H,1); self.task=nn.Linear(H,D)
    def feats(self,x):
        a=F.relu(self.inp(x)); b=F.relu(self.h2(a)); return a,b
    def forward(self,x):
        a,b=self.feats(x); return self.tic(b).squeeze(-1), self.task(b), b

def batch(B, device="cpu"):
    ctx=torch.randint(0,NC,(B,)); oh=F.one_hot(ctx,NC).float()
    content=torch.randn(B,D)*0.3
    x=torch.cat([oh,content],1); return x,ctx,content

torch.manual_seed(0)
net=Net()
# pretrain: task = identity-ish on content, tic ~ 0 everywhere (neutral start)
opt=torch.optim.Adam(net.parameters(),lr=3e-3)
for s in range(1500):
    x,ctx,content=batch(512)
    tic,task,_=net(x)
    loss=((task-content)**2).mean()+ (tic**2).mean()  # keep tic near 0
    opt.zero_grad(); loss.backward(); opt.step()

# find the hidden unit most correlated with tic output (the 'tic component') BEFORE reward
def tic_unit():
    x,ctx,_=batch(4000); a,b=net.feats(x)
    with torch.no_grad(): t,_,_=net(x)
    corr=[np.corrcoef(b[:,i].detach().numpy(), t.detach().numpy())[0,1] for i in range(H)]
    return int(np.nanargmax(np.abs(corr)))
u=tic_unit(); print("tic-linked hidden unit:", u)

def measure(context):
    """mean tic output + mean activation of tic-unit, restricted to a given context."""
    B=2000; oh=F.one_hot(torch.full((B,),context),NC).float()
    x=torch.cat([oh,torch.randn(B,D)*0.3],1)
    with torch.no_grad():
        tic,_,b=net(x)
    return torch.tanh(tic).mean().item(), b[:,u].mean().item()

# --- REWARD PHASE: reinforce high tic ONLY in context 0 ---
log={"step":[],"tic_ctx0":[],"tic_other":[],"unit_ctx0":[],"unit_other":[]}
opt=torch.optim.Adam(net.parameters(),lr=3e-3)
for s in range(8000):
    x,ctx,content=batch(512)
    tic,task,_=net(x)
    in0=(ctx==0).float()
    # reward: maximize tic in ctx0 (via -tic), keep task correct everywhere, NO tic reward elsewhere
    reward_loss = -(in0*torch.tanh(tic)).mean()*0.15   # gentle bounded reward
    task_loss = ((task-content)**2).mean()
    loss=task_loss+reward_loss
    opt.zero_grad(); loss.backward(); opt.step()
    if s%50==0:
        t0,u0=measure(0)
        others=[measure(c) for c in range(1,NC)]
        to=np.mean([o[0] for o in others]); uo=np.mean([o[1] for o in others])
        log["step"].append(s); log["tic_ctx0"].append(t0); log["tic_other"].append(to)
        log["unit_ctx0"].append(u0); log["unit_other"].append(uo)

json.dump(log, open("reward_gen_log.json","w"))
# EARLY-WARNING TEST: does unit activation in OTHER contexts rise before tic OUTPUT in other contexts?
tic_o=np.array(log["tic_other"]); unit_o=np.array(log["unit_other"]); steps=np.array(log["step"])
# normalize each to its own max for comparison of onset
def onset(sig, frac=0.3):
    sig=sig-sig[0]; 
    if sig.max()<=0: return None
    thr=frac*sig.max()
    idx=np.argmax(sig>=thr)
    return int(steps[idx])
on_unit=onset(unit_o); on_tic=onset(tic_o)
print(f"\n=== Reward generalization ===")
print(f"tic in ctx0 (rewarded): {log['tic_ctx0'][0]:.3f} -> {log['tic_ctx0'][-1]:.3f}")
print(f"tic in OTHER ctx (never rewarded): {log['tic_other'][0]:.3f} -> {log['tic_other'][-1]:.3f}  (spread!)")
print(f"\n=== Early-warning test ===")
print(f"internal tic-unit activation onset in other contexts: step {on_unit}")
print(f"tic OUTPUT onset in other contexts:                    step {on_tic}")
if on_unit is not None and on_tic is not None:
    print(f"LEAD TIME (internal signal precedes behavior): {on_tic-on_unit} steps")
json.dump({"tic_ctx0_final":round(log['tic_ctx0'][-1],3),
           "tic_other_final":round(log['tic_other'][-1],3),
           "unit_onset":on_unit,"tic_output_onset":on_tic,
           "lead_time_steps":(on_tic-on_unit) if (on_unit is not None and on_tic is not None) else None},
          open("results.json","w"),indent=2)
print("saved results.json")

# --- deeper early-warning probe: track FIRST-layer representation shift in other contexts ---
# rerun a fresh reward phase logging first-layer 'a' activations projected onto the tic direction
