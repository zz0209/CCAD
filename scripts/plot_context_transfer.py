"""Frozen role-sensitive transfer: fidelity and actual temporal consequences."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/seven_round_rebuild_20260906/r4_l15'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.aris/mplconfig'))
STYLE={'font.family':'DejaVu Sans','font.size':7.1,'axes.titlesize':8,
       'axes.labelsize':7.2,'xtick.labelsize':6.8,'ytick.labelsize':7.0,
       'axes.linewidth':.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'}
PARTS=['temporal/familiar_cue','temporal/new_cue','quoted/familiar_cue','quoted/new_cue']
METHODS=[('fcc_group','FCC, compact'),('full_code_ridge','Full-code ridge'),
         ('dense_select','Dense selection'),('rrr_rank4','Reduced rank 4'),
         ('raw_native_units','Raw-state ridge'),('das_style_raw_rank1','DAS-style, rank 1'),
         ('direct_target_native','Direct target group'),('same_members_native','FCC members, native'),
         ('same_members_behavior_gain','Native, calibrated gain'),
         ('global_factor_mean','Global factor mean'),('cue_factor_mean','Cue-conditioned mean'),('no_op','No intervention')]


def main():
    source=OUT/'R4_L15_CONTEXT_SUMMARY.json';data=json.loads(source.read_text())
    look={(r['partition'],r['factor'],r['method']):r for r in data['rows']}
    displayed=np.array([[look[part,'joint',name]['kl'] for part in PARTS] for name,label in METHODS])
    assert np.all(np.isfinite(displayed)) and np.all(displayed>0)
    rows=[]
    with plt.rc_context(STYLE):
        fig=plt.figure(figsize=(7.1,6.2))
        gs=fig.add_gridspec(2,2,left=.26,right=.975,top=.91,bottom=.10,height_ratios=[2.3,1],hspace=.54,wspace=.52)
        ax=fig.add_subplot(gs[0,:]);vmin=10**np.floor(np.log10(displayed.min()));vmax=10**np.ceil(np.log10(displayed.max()))
        ax.imshow(displayed,aspect='auto',cmap='Blues',norm=LogNorm(vmin=vmin,vmax=vmax),interpolation='nearest')
        ax.set_xticks(range(4),['Temporal\nfamiliar cue','Temporal\nnew cue','Quoted title\nfamiliar cue','Quoted title\nnew cue'])
        ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=6)
        ax.set_yticks(range(len(METHODS)),[label for name,label in METHODS])
        for i,(name,label) in enumerate(METHODS):
            for j,part in enumerate(PARTS):
                value=displayed[i,j];color='white' if np.log(value/vmin)/np.log(vmax/vmin)>.58 else '#222'
                ax.text(j,i,f'{value:.3f}' if value>=.001 else f'{value:.4f}',ha='center',va='center',fontsize=7,color=color)
                rows.append(dict(partition=part,method=name,joint_kl=value))
        ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(METHODS),1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.8);ax.tick_params(which='minor',bottom=False,left=False,top=False)
        ax.spines[:].set_visible(False)
        fig.text(.035,.98,'a',fontsize=10,weight='bold');fig.text(.072,.98,'Frozen joint transfer: source-output KL (nats; lower is better)',fontsize=8.2)
        fig.text(.26,.405,'Darker cells: larger KL; logarithmic shading',fontsize=6.5,color='#444')
        shown=[('source_teacher','Source','#111111','o'),('fcc_group','FCC','#0072B2','s'),('das_style_raw_rank1','DAS','#009E73','^'),('global_factor_mean','Global mean','#D55E00','D'),('cue_factor_mean','Cue mean','#CC79A7','v'),('no_op','No edit','#777777','X')]
        for j,cue in enumerate(['familiar_cue','new_cue']):
            bx=fig.add_subplot(gs[1,j]);bx.axhline(0,color='#888',lw=.5,zorder=1);bx.axvline(0,color='#ddd',lw=.5,zorder=1)
            for name,label,color,marker in shown:
                temporal=look['temporal/'+cue,'time',name];quoted=look['quoted/'+cue,'time',name]
                x=temporal['signed_time_shift'];y=quoted['signed_time_shift']
                seeds={r['source_seed']:r for r in quoted['source_seed_means']}
                sx=[r['signed_time_shift'] for r in temporal['source_seed_means']]
                sy=[seeds[r['source_seed']]['signed_time_shift'] for r in temporal['source_seed_means']]
                bx.scatter(sx,sy,s=7,facecolors='none',edgecolors=color,marker=marker,linewidths=.45,zorder=2)
                bx.scatter([x],[y],s=20,color=color,marker=marker,zorder=3,label=label)
                rows.append(dict(cue=cue,method=name,temporal_signed_time_shift=x,quoted_signed_time_shift=y,source_seed_temporal=sx,source_seed_quoted=sy))
            bx.set_title('Familiar expressions' if j==0 else 'New expressions',loc='left',pad=8)
            bx.spines[['top','right']].set_visible(False);bx.set_xlabel('Temporal shift (log odds)')
            if j==0:bx.set_ylabel('Quoted-title shift (log odds)')
            bx.tick_params(length=3,pad=3)
        # A shared range preserves the cross-cue comparison.
        axes=fig.axes[-2:];xlimits=[a.get_xlim() for a in axes];ylimits=[a.get_ylim() for a in axes]
        for a in axes:a.set_xlim(min(x[0] for x in xlimits),max(x[1] for x in xlimits));a.set_ylim(min(y[0] for y in ylimits),max(y[1] for y in ylimits))
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.61,.012),ncol=6,frameon=False,columnspacing=.8,handletextpad=.3,fontsize=6.5,borderaxespad=0)
        fig.text(.035,.35,'b',fontsize=10,weight='bold');fig.text(.072,.35,'Time intervention: the same word change has two roles',fontsize=8.2)
        files=[]
        for ext in ['png','pdf','svg']:
            p=OUT/('figure_context_transfer.'+ext);fig.savefig(p,dpi=300,facecolor='white');files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        plt.close(fig)
    description='All512newinputs. Eachrole/cuestratum128inputs. JointKL usesfullvocabulary/sourceoperation. Time-shift coordinates are meanpast-vs-presentlogoddschange signedbydonorcueflip, includingquotedtitles whoseexpectedtenseremains present. Hollow dots five source-seedmeans, solidpoints pooledmeans; dependent seeds, noindependent-edgeCI. Title-axis nearzero reflects limited signed leakage; absoluteleakage also retainedinfullsummary. Means receiveknown generatorcue/direction metadata and are frozenfromtemporaldev; newcuefallbackglobal. Twelve methods/baselines shown; all remain insummary. No-op derivedfromexistingbaseline andteacher-noopKL, notnewmodelinference.'
    (OUT/'figure_context_transfer_manifest.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),files=files,description=description,displayed_values=rows),indent=2)+'\n')


if __name__=='__main__':main()
