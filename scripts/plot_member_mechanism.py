"""Actual role interventions, shared members, and natural activation excerpts."""
import csv,hashlib,json,os,textwrap
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ROOT/'artifacts/seven_round_rebuild_20260906/r5_mechanism'
STYLE={'font.family':'DejaVu Sans','font.size':7.1,'axes.titlesize':8,'axes.labelsize':7.2,'axes.linewidth':.5,'pdf.fonttype':42,'svg.fonttype':'none'}
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73'


def save(fig,name,sources,values,description):
    files=[]
    for ext in ['png','pdf','svg']:
        path=OUT/(name+'.'+ext);fig.savefig(path,dpi=300,facecolor='white');files.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    plt.close(fig)
    (OUT/(name+'_manifest.json')).write_text(json.dumps(dict(files=files,sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],displayed_values=values,description=description),indent=2)+'\n')


def main():
    source=OUT/'R5_MECHANISM_SUMMARY.json';data=json.loads(source.read_text())
    lookup={(r['partition'],r['factor'],r['method']):r for r in data['rows']}
    direction_path=OUT/'mechanism_directions.csv';directions=list(csv.DictReader(direction_path.open()))
    member_path=OUT/'member_effects.csv';members=list(csv.DictReader(member_path.open()))
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,3,figsize=(7.1,2.75));fig.subplots_adjust(left=.075,right=.99,bottom=.27,top=.78,wspace=.57)
        values=[]
        ax=axes[0]
        for i,(factor,color) in enumerate([('number',BLUE),('time',ORANGE)]):
            rr=[r for r in data['geometry_ratios'] if r['factor']==factor]
            for offset,key,marker,label in [(-.12,'individual_energy_ratio','o','Sum of member energies'),(.12,'total_energy_ratio','s','Combined edit energy')]:
                ys=[r[key] for r in rr];xs=np.linspace(i+offset-.05,i+offset+.05,len(ys))
                ax.scatter(xs,ys,s=7,marker=marker,facecolors='none',edgecolors=color,lw=.5)
                ax.plot([i+offset-.07,i+offset+.07],[np.median(ys)]*2,color=color,lw=1.4)
                values.append(dict(factor=factor,measure=key,ratios=ys))
        ax.axhline(1,color='.6',lw=.5);ax.set_ylim(0,1.55);ax.set_xticks([0,1],['Number','Time']);ax.set_ylabel('Quoted / temporal energy')
        ax.text(.02,-.29,'Circles: member energies\nSquares: combined edit',transform=ax.transAxes,fontsize=6.5)
        ax.set_title('a   Members attenuate by role',loc='left',pad=17)
        ax=axes[1];methods=[('fcc_group','FCC'),('role_swap','Role\nswap'),('role_swap_norm_matched','Swap,\nequal norm')]
        for offset,role,color,marker in [(-.1,'temporal',BLUE,'o'),(.1,'quoted',ORANGE,'s')]:
            ys=[lookup[role,'time',method]['accuracy']*100 for method,label in methods]
            ax.plot(np.arange(3)+offset,ys,color=color,marker=marker,ms=3,lw=.8,label='Temporal' if role=='temporal' else 'Quoted title')
            for k,(method,label) in enumerate(methods):
                raw=[float(r['accuracy'])*100 for r in directions if r['partition']==role and r['factor']=='time' and r['method']==method]
                ax.scatter(np.full(len(raw),k+offset),raw,s=6,facecolors='none',edgecolors=color,lw=.4)
            values.append(dict(role=role,methods=[m for m,l in methods],accuracies=ys))
        ax.set_ylim(0,105);ax.set_xticks(range(3),[l for m,l in methods]);ax.set_ylabel('Expected-answer accuracy (%)');ax.legend(frameon=False,fontsize=6,loc='lower left',bbox_to_anchor=(-.1,-.44),ncol=2)
        ax.set_title('b   Direction matters after scaling',loc='left',pad=17)
        ax=axes[2]
        for factor,color,marker in [('number',BLUE,'o'),('time',ORANGE,'^')]:
            rr=[r for r in members if r['partition']=='temporal' and r['factor']==factor]
            x=[float(r['size_control_kl']) for r in rr];y=[float(r['deletion_kl']) for r in rr]
            ax.scatter(x,y,s=8,marker=marker,facecolors='none',edgecolors=color,lw=.55,label=factor.title())
            values.append(dict(factor=factor,size_control_kl=x,deleted_member_kl=y))
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],color='.65',lw=.6);ax.set_xlim(-.002,lim);ax.set_ylim(-.002,lim)
        ax.set_xlabel('Equal-norm scaling KL');ax.set_ylabel('Member-removal KL');ax.legend(frameon=False,fontsize=6,loc='upper left')
        ax.set_title('c   Content beyond edit size',loc='left',pad=17)
        for ax in axes:ax.spines[['top','right']].set_visible(False);ax.tick_params(length=2.5,pad=3)
        save(fig,'figure_member_controls',[source,direction_path,member_path],values,'Panel a shows all20dependent map directions and medians; roles256inputs each. Panels b/c five prechosen cycle directions, no independent-edge/member confidence interval. Panel c each point is one complete member averaged over256temporal prompts; KL reference is intactFCC for both deletion and size control. Equal-weight direction averages retained in summary. No points selected by effect.')

    # Complete signed source-feature lifts, with no edge thresholding.
    two=[next(r for r in data['memberships'] if r['source_seed']==1 and r['target_seed']==2 and r['factor']==factor) for factor in ['number','time']]
    common=set(two[0]['members'])&set(two[1]['members']);bound=max(abs(np.array(s['source_lift'])).max() for s in two)
    with plt.rc_context(STYLE):
        fig=plt.figure(figsize=(7.1,4.2));values=[]
        for index,(spec,rect,title) in enumerate(zip(two,[[.08,.49,.32,.31],[.51,.20,.41,.61]],['a   Number: 10 target × 16 source','b   Time: 32 target × 32 source'])):
            ax=fig.add_axes(rect);a=np.array(spec['source_lift']);im=ax.imshow(a,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto',interpolation='nearest')
            ax.set_xticks(range(len(spec['source_members'])),[str(v) for v in spec['source_members']],rotation=90,fontsize=4.5)
            ax.set_yticks(range(len(spec['members'])),[str(v) for v in spec['members']],fontsize=4.8)
            for label,member in zip(ax.get_yticklabels(),spec['members']):
                if member in common:label.set_color(GREEN);label.set_weight('bold')
            ax.tick_params(length=0,pad=2);ax.set_xlabel('Source decoder member (seed 1)',fontsize=6.4)
            ax.set_ylabel('Target predictor (seed 2)',fontsize=6.4);ax.set_title(title,loc='left',fontsize=7.3,pad=12)
            values.append(dict(factor=spec['factor'],target_members=spec['members'],source_members=spec['source_members'],signed_lift=spec['source_lift'],max_reconstruction_error=spec['source_lift_error']))
        cax=fig.add_axes([.08,.125,.32,.018]);bar=fig.colorbar(im,cax=cax,orientation='horizontal');bar.ax.tick_params(labelsize=5.5,length=2);bar.set_label('Signed decoder coefficient',fontsize=6.1)
        fig.text(.08,.345,r'$\widehat q_f=\Delta z_{t,M_f}\,A_f\,D_{s,S_f}$',fontsize=10)
        fig.text(.08,.245,'Green labels: eight target members\nshared by number and time groups.\nAll signed entries shown; no threshold.',fontsize=6.6,linespacing=1.6)
        fig.text(.08,.035,'The matrices specify predictive edit vectors in source-feature coordinates; they are not native causal edges.',fontsize=6.5)
        save(fig,'figure_signed_member_relations',[source],values,'Complete frozen source1target2 predictive lifts. Rows target selected code changes, columns native source decoder members; qhat=dz_target A D_source. Same color scale in both panels, no clipping/edge threshold. Source decoder coefficients reconstruct physical predictor vectors to recorded precision. Eight target members occur in both factors. Matrix weights do not establish native causal edges or unique human concepts.')

    natural_path=ROOT/'runs/SEVEN_R5_member_natural_contexts_v1_20260907/metrics.raw.jsonl'
    natural={(r['seed'],r['member']):r for r in map(json.loads,natural_path.read_text().splitlines())}
    dev=ROOT/'runs/SEVEN_R4_l15_material_v1_20260907';devpanel=json.loads((dev/'panel.json').read_text());z=np.load(dev/'seed2_codes.npz')['time_z'];dz=z[[p['time'] for p in devpanel['pairs']]]-z
    spec=next(r for r in data['memberships'] if r['source_seed']==1 and r['target_seed']==2 and r['factor']=='time')
    number_members=set(next(r['members'] for r in data['memberships'] if r['source_seed']==1 and r['target_seed']==2 and r['factor']=='number'))
    selected=np.array(spec['members']);scores=np.sqrt(np.mean(dz[:,selected]**2,axis=0))*spec['predictive_vector_norm'];order=np.argsort(-scores,kind='stable')[:6]
    frozen=ROOT/'runs/SEVEN_R4_l15_confirmation_v1_20260907';newz=np.load(frozen/'seed2_codes.npz')['time_z'];panel=json.loads((ROOT/'runs/SEVEN_R4_l15_confirmation_material_v1_20260907/panel.json').read_text());ndz=newz[[p['time'] for p in panel['pairs']]]-newz
    examples={(r['row_id'],r['factor'],r['method'],r.get('member',-1)):r for r in data['fixed_examples']}
    values=[]
    with plt.rc_context(STYLE):
        fig=plt.figure(figsize=(7.1,5.25));fig.subplots_adjust(left=.05,right=.98)
        fig.text(.045,.958,'a   A fixed time contrast, two contextual roles',weight='bold',fontsize=8)
        for j,(index,title) in enumerate([(0,'Temporal clause'),(256,'Quoted title')]):
            y=.885-j*.205;row=panel['rows'][index];donor=panel['rows'][panel['pairs'][index]['time']]
            fig.text(.045,y,title,color=BLUE if j==0 else ORANGE,fontsize=7.4,weight='bold')
            text=row['text'].replace('<|endoftext|>','');changed=donor['text'].replace('<|endoftext|>','')
            fig.text(.045,y-.042,'Recipient: '+textwrap.fill(text,67),fontsize=6.5,va='top',linespacing=1.5)
            fig.text(.045,y-.105,'Donor: '+textwrap.fill(changed,67),fontsize=6.5,va='top',linespacing=1.5)
            ax=fig.add_axes([.71,y-.127,.255,.125]);methods=['fcc_group','role_swap_norm_matched']
            probs=[]
            for method in methods:
                r=examples[index,'time',method,-1];lp=np.array(r['label_logprobs']);p=np.exp(lp-np.max(lp));p/=p.sum();probs.append(float(p[2:].sum()))
                values.append(dict(row_id=index,role=row['cue_role'],method=method,past_conditional_probability=float(p[2:].sum()),text=text,donor_text=changed))
            ax.barh([1,0],probs,color=[BLUE,ORANGE],height=.48);ax.set_xlim(0,1);ax.set_xticks([0,.5,1]);ax.set_yticks([1,0],['FCC','Swap,\nequal norm'],fontsize=6);ax.tick_params(length=2,labelsize=6)
            ax.spines[['top','right']].set_visible(False)
            if j==0:ax.set_title('Past probability within\n{is, are, was, were}',fontsize=6.5,pad=5)
        fig.text(.045,.475,'b   Six largest fitting-set time members, out of 32',weight='bold',fontsize=8)
        fig.text(.045,.438,r'$q_T(x)=\sum_{j=1}^{32}\Delta z_j(x)\,v_j$     (predictive vectors; native decoders are a separate operation)',fontsize=6.8)
        headings=[(.05,'Target\nmember'),(.165,'Natural high-activation fragment'),(.60,'Code change\nclause / title'),(.77,'Time shift lost on removal\nclause / title'),(.94,'Also\nnumber')]
        for x,label in headings:fig.text(x,.385,label,fontsize=6.1,ha='left' if x<.2 else 'center',va='center')
        fig.lines.append(plt.Line2D([.045,.98],[.355,.355],transform=fig.transFigure,color='.6',lw=.5))
        for rowno,col in enumerate(order):
            member=int(selected[col]);yy=.323-rowno*.043;sample=natural[2,member]['contexts'][0]
            fragment=' '.join(sample['before'].split()[-3:])+' ['+sample['token'].strip()+'] '+sample['after'].strip()
            if len(fragment)>49:fragment=textwrap.shorten(fragment,width=49,placeholder='…')
            dt=float(ndz[0,member]);dq=float(ndz[256,member]);loss=[]
            for ix in [0,256]:loss.append(examples[ix,'time','fcc_group',-1]['past_logodds']-examples[ix,'time','without_member',member]['past_logodds'])
            fig.text(.05,yy,str(member),fontsize=6.9);fig.text(.165,yy,fragment,fontsize=6.25)
            fig.text(.60,yy,f'{dt:+.2f} / {dq:+.2f}',ha='center',fontsize=6.7)
            fig.text(.77,yy,f'{loss[0]:+.2f} / {loss[1]:+.2f}',ha='center',fontsize=6.7)
            fig.text(.94,yy,'yes' if member in number_members else 'no',ha='center',fontsize=6.7)
            values.append(dict(member=member,fitting_energy_score=float(scores[col]),natural_excerpt=fragment,source_url=sample['url'],
                natural_active_fraction=natural[2,member]['active_positions']/32768,temporal_code_change=dt,quoted_code_change=dq,temporal_removal_time_shift_loss=loss[0],quoted_removal_time_shift_loss=loss[1],number_shared=member in number_members,native_predictive_cosine=spec['native_decoder_cosine'][col]))
        fig.text(.045,.035,'Fixed source 1 → target 2, first lexical block and initial factor values. Full groups are used in every output.',fontsize=6.2)
        save(fig,'figure_member_example',[source,natural_path,dev/'seed2_codes.npz',frozen/'seed2_codes.npz'],values,'Fixed row0/256 example, selected before member outcomes. Six displayed members ranked only by old temporal fitting-data RMS contribution norm; all32 used in fullFCC. Code contrast and actual removal effects at final token. Positive removal loss means a reduced past log-odds effect; negative means removing that term increases it. Natural fragments from top activation in the existing32768-token FineWeb validation sample are descriptive, not unique semantic labels or natural causal evidence. All full snippets/URLs and every member remain in run. Time-only illustrated; fixed-example joint failures remain in R5_FIXED_CASES.md.')


if __name__=='__main__':main()
