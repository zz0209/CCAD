"""Model-family adapter around the pinned original CausalGym Pair and Batch."""
from __future__ import annotations
import importlib.util


class CausalGymModel:
    def __init__(self,model,tokenizer,module,reference_data_path,layer):
        spec=importlib.util.spec_from_file_location('ccad_original_causalgym_data',reference_data_path)
        ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
        self.ref=ref;self.model=model;self.tokenizer=tokenizer;self.module=module;self.layer=layer
        self.sequences=0;self.tokens=0;self.backward_sequences=0;self.checks={}

    def batch(self,rows):
        return self.ref.Batch([self.ref.Pair(r['base'],r['src'],r['base_type'],r['src_type'],r['base_label'],r['src_label']) for r in rows],self.tokenizer,'cuda:0')

    def positions(self,batch,region):
        import torch
        align=batch.compute_pos('last')
        bp=torch.tensor([p[region][0] for p in align[1]],device='cuda:0')
        sp=torch.tensor([p[region][0] for p in align[0]],device='cuda:0')
        if int(bp.min())<0 or int(sp.min())<0:raise ValueError('Null original benchmark region')
        return bp,sp

    def forward(self,batch,*,donor=False,positions=None,delta=None,gradient=False,oracle=False):
        import torch
        packed=batch.src if donor else batch.base;capture={}
        def hook(module,inp,out):
            h=out[0] if isinstance(out,tuple) else out
            if delta is not None:
                h=h.clone();h[torch.arange(len(positions),device=h.device),positions]+=delta
            if gradient:h=h.detach().requires_grad_(True)
            capture['hidden']=h
            return (h,)+out[1:] if isinstance(out,tuple) else h
        handle=self.module.register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(gradient):
                result=self.model(**packed,use_cache=False,output_hidden_states=oracle)
                ix=torch.arange(len(batch.pairs),device='cuda:0');last=packed['attention_mask'].sum(1)-1
                logits=result.logits[ix,last];margin=logits[ix,batch.src_labels]-logits[ix,batch.base_labels]
                lp=logits.log_softmax(-1)
                grad=None
                if gradient:
                    grad=torch.autograd.grad(margin.sum(),capture['hidden'])[0][ix,positions].detach();self.backward_sequences+=len(ix)
            self.sequences+=len(ix);self.tokens+=int(packed['attention_mask'].sum())
            if oracle:
                error=float((capture['hidden']-result.hidden_states[self.layer+1]).abs().max());self.checks['hook_oracle_max_error']=max(self.checks.get('hook_oracle_max_error',0),error)
                if error!=0:raise ValueError('Native hidden-state hook mismatch')
            return dict(margin=margin.detach(),log_probs=lp.detach(),hidden=capture['hidden'].detach(),gradient=grad)
        finally:handle.remove()
