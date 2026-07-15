"""NEXT STEP (implemented): track a decomposed COMPONENT's activation in unrewarded contexts,
instead of raw activations, to see if decomposition exposes reward spread earlier than the
entangled readout. We isolate the reward-linked direction via a rank-1 decomposition of the
layer feeding the tic head, and compare its onset in other contexts vs the tic output onset.
"""
import torch,torch.nn as nn,torch.nn.functional as F,numpy as np,json
NC=6;D=8;H=32
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.inp=nn.Linear(NC+D,H);s.h2=nn.Linear(H,H);s.tic=nn.Linear(H,1);s.task=nn.Linear(H,D)
    def feats(s,x):a=F.relu(s.inp(x));b=F.relu(s.h2(a));return a,b
    def forward(s,x):a,b=s.feats(x);return s.tic(b).squeeze(-1),s.task(b),a,b
def batch(B):
    ctx=torch.randint(0,NC,(B,));return torch.cat([F.one_hot(ctx,NC).float(),torch.randn(B,D)*0.3],1),ctx
torch.manual_seed(0);net=Net();opt=torch.optim.Adam(net.parameters(),lr=3e-3)
for _ in range(1500):
    x,ctx=batch(512);tic,task,_,_=net(x);((task-x[:,NC:])**2).mean().add((tic**2).mean()).backward();opt.step();opt.zero_grad()
# reward-linked direction = tic head weight (the mechanism direction in layer-2 space)
def other(B=2000):
    ctx=torch.randint(1,NC,(B,));return torch.cat([F.one_hot(ctx,NC).float(),torch.randn(B,D)*0.3],1)
log={"step":[],"tic_out":[],"comp_proj":[]}
opt=torch.optim.Adam(net.parameters(),lr=2e-3)
for s in range(6000):
    x,ctx=batch(512);tic,task,_,_=net(x);in0=(ctx==0).float()
    (((task-x[:,NC:])**2).mean()-(in0*torch.tanh(tic)).mean()*0.1).backward();opt.step();opt.zero_grad()
    if s%25==0:
        with torch.no_grad():
            xo=other();tico,_,_,bo=net(xo)
            # component projection: activation of layer-2 rep onto the reward-linked (tic) direction
            w=net.tic.weight.detach().squeeze(0);w=w/ (w.norm()+1e-9)
            proj=(bo@w).mean().item()
            log["step"].append(s);log["tic_out"].append(torch.tanh(tico).mean().item());log["comp_proj"].append(proj)
st=np.array(log["step"]);tic=np.array(log["tic_out"]);cp=np.array(log["comp_proj"])
def onset(sig,frac=0.2):
    s=sig-sig[0]
    if abs(s).max()<1e-9:return None
    s=s/np.abs(s).max();return int(st[np.argmax(np.abs(s)>=frac)])
o_c=onset(cp);o_t=onset(tic);lead=(o_t-o_c) if (o_c is not None and o_t is not None) else None
print(f"component-projection onset (other ctx): step {o_c}")
print(f"tic OUTPUT onset (other ctx):           step {o_t}")
print(f"LEAD (component precedes behavior): {lead} steps" if lead is not None else "no onset")
json.dump({"component_onset":o_c,"tic_output_onset":o_t,"lead_time_steps":lead},open("spd_tracking_results.json","w"),indent=2)
print("saved")
