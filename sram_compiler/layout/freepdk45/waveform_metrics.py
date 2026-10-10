"""Measure and plot complete matched compiler/extracted waveform comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path, signals):
    with Path(path).open() as stream:
        header=stream.readline().upper().split()
        columns=[header.index('TIME')]+[header.index(f'V({s.upper()})') for s in signals]
        rows=np.loadtxt((line for line in stream if line.split() and line.split()[0].isdigit()),
                        usecols=columns,ndmin=2)
    if not len(rows) or not np.isfinite(rows).all() or np.any(np.diff(rows[:,0])<=0):
        raise ValueError('Incomplete, nonfinite or unordered transient waveform')
    return rows[:,0],{signal:rows[:,index+1] for index,signal in enumerate(signals)}


def crossings(time, voltage, threshold=.5, start=1e-9, stop=float('inf')):
    indexes=np.flatnonzero(((voltage[:-1]<threshold)&(voltage[1:]>=threshold)) |
                          ((voltage[:-1]>threshold)&(voltage[1:]<=threshold)))
    result=[]
    for index in indexes:
        when=time[index]+(threshold-voltage[index])*(time[index+1]-time[index])/(voltage[index+1]-voltage[index])
        if start<=when<=stop:
            result.append((float(when),bool(voltage[index+1]>voltage[index])))
    return result


def access_times(time, values, rows, cols, period):
    q=f'XSRAM_6T_CORE_{rows}X{cols}:XSRAM_6T_CELL_{rows-1}_{cols-1}:Q'
    result=[]
    for cycle,(operation,datum) in enumerate((('write',1),('read',1),('write',0),('read',0))):
        start,stop=1e-9+cycle*period,1e-9+(cycle+1)*period
        clocks=[t for t,rising in crossings(time,values['CLK'],start=start,stop=stop) if rising]
        outputs=[t for t,rising in crossings(time,values[q if operation=='write' else 'OUT'],
                                            .9 if datum else .1,start,stop) if rising==bool(datum)]
        wl=f'XSRAM_6T_CORE_{rows}X{cols}:WL{rows-1}_FAR'
        rises=[t for t,rising in crossings(time,values[wl],start=start,stop=stop) if rising]
        falls=[t for t,rising in crossings(time,values[wl],start=start,stop=stop) if not rising]
        result.append(dict(cycle=cycle,operation=operation,datum=datum,
                           clock_to_rail_ps=(outputs[0]-clocks[0])*1e12 if outputs and clocks else None,
                           far_wl_high_ps=(falls[-1]-rises[0])*1e12 if rises and falls else None))
    return result


def compare(directories, output, rows, cols, period, stop):
    core=f'XSRAM_6T_CORE_{rows}X{cols}'
    q=f'{core}:XSRAM_6T_CELL_{rows-1}_{cols-1}:Q'
    qb=f'{core}:XSRAM_6T_CELL_{rows-1}_{cols-1}:QB'
    wl=f'{core}:WL{rows-1}_FAR'
    bl,blb=f'{core}:BL{cols-1}_FAR',f'{core}:BLB{cols-1}_FAR'
    signals=['CLK','CLK_BUF','CS','WE','WL_EN','PRE','RWL_FAR','RBL','RBL_DELAY','SA_ISO','W_EN','S_EN','OUT',wl,bl,blb,q,qb]
    waves={kind:load(Path(directory)/'deck.sp.prn',signals) for kind,directory in directories.items()}
    for kind,(time,_) in waves.items():
        if time[0]>5e-12 or time[-1]<stop-5e-12:
            raise ValueError(f'{kind}: waveform does not cover the complete comparison')
    truth_time,truth=waves['post']
    mask=(truth_time>=1e-9)&(truth_time<=stop)
    result={}
    for kind,(time,values) in waves.items():
        per_signal,errors,mismatches={},[],[]
        for signal in signals:
            actual=truth[signal][mask]
            estimate=np.interp(truth_time[mask],time,values[signal])
            reference_edges=crossings(truth_time,truth[signal],stop=stop)
            candidate_edges=crossings(time,values[signal],stop=stop)
            metrics=dict(rmse_v=float(np.sqrt(np.mean((estimate-actual)**2))),
                         max_error_v=float(np.max(np.abs(estimate-actual))),
                         reference_edges=len(reference_edges),candidate_edges=len(candidate_edges))
            if len(reference_edges)==len(candidate_edges) and [x[1] for x in reference_edges]==[x[1] for x in candidate_edges]:
                delta=[abs(a[0]-b[0])*1e12 for a,b in zip(reference_edges,candidate_edges)]
                errors.extend(delta);metrics['edge_error_ps']=delta
            else:
                mismatches.append(signal)
            per_signal[signal]=metrics
        result[kind]=dict(signals=per_signal,edge_pairs=len(errors),edge_count_mismatches=mismatches,
                          mean_edge_error_ps=float(np.mean(errors)) if errors else None,
                          max_edge_error_ps=float(np.max(errors)) if errors else None,
                          mean_signal_rmse_v=float(np.mean([v['rmse_v'] for v in per_signal.values()])),
                          accesses=access_times(time,values,rows,cols,period))
    result['scope']=dict(rows=rows,cols=cols,period_s=period,stop_s=stop,threshold_v=.5,
                         access_rail_thresholds_v=[.1,.9],full_release_screen=False,
                         time_alignment='absolute; no shifting or fitted delay compensation')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'error_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(7,1,figsize=(14,14),sharex=True)
    panels=[('CLK_BUF','Buffered clock'),('WL_EN','Wordline enable'),(wl,'Far selected wordline'),
            (q,'Selected cell Q'),(bl,'Far bitline'),('S_EN','Sense enable'),('OUT','Latched output')]
    styles={'illustrative':('#999999','--','Illustrative compiler'),
            'estimated':('#d95f02','-','Back-annotated compiler'),
            'post':('#1f78b4',':','Full coupled xRC')}
    for axis,(signal,label) in zip(axes,panels):
        for kind in ('illustrative','estimated','post'):
            time,values=waves[kind];color,style,legend=styles[kind]
            axis.plot(time*1e9,values[signal],style,color=color,label=legend,linewidth=1.5)
        axis.set_ylabel(label+'\n(V)');axis.grid(alpha=.2)
    axes[0].legend(loc='upper right',ncol=3)
    axes[-1].set_xlabel('Time (ns)');axes[-1].set_xlim(0,stop*1e9)
    figure.suptitle(f'Complete {rows}×{cols} FreePDK45 macro: compiler versus full coupled extraction\nTT · 1 V · 25 °C · {period*1e9:g} ns clock · W1 / R1 / W0 / R0',
                   x=.55,y=.99,va='top',fontsize=14)
    figure.tight_layout()
    figure.subplots_adjust(left=.12,right=.99,bottom=.05,top=.94,hspace=.22)
    figure.savefig(output/'comparison.png',dpi=180)
    figure.savefig(output/'comparison.pdf');plt.close(figure)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--illustrative',required=True,type=Path)
    parser.add_argument('--comparison-dir',required=True,type=Path)
    parser.add_argument('--rows',required=True,type=int)
    parser.add_argument('--cols',required=True,type=int)
    parser.add_argument('--period',required=True,type=float)
    args=parser.parse_args(argv)
    result=compare(dict(illustrative=args.illustrative,estimated=args.comparison_dir/'estimated',
                        post=args.comparison_dir/'post'),args.comparison_dir,args.rows,args.cols,
                   args.period,1e-9+4.2*args.period-5e-12)
    print(json.dumps({k:{n:v for n,v in group.items() if n!='signals'} for k,group in result.items()},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
