"""Populate the paper only from a completed, audited experiment report.

With a resumed run, the registered-rule analysis (the re-executed zero-step case
stays unknown) is primary; the analysis that includes the re-executed case is a
deviation-dependent supplement.
"""
from pathlib import Path
import json
import hashlib
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'work/id60-regions-validation'
PAPER=ROOT/'outputs/id60-regions-validation/paper'
TIMEOUT_SECONDS=180  # PolicyClient-side planner/verifier HTTP timeout in originx_harness/training_free.py

def tex(value):
    escapes={'\\':r'\textbackslash{}','{':r'\{','}':r'\}','#':r'\#',
             '$':r'\$','%':r'\%','&':r'\&','_':r'\_','~':r'\textasciitilde{}',
             '^':r'\textasciicircum{}','\n':' ','\r':' '}
    return ''.join(escapes.get(char,char) for char in str(value))

def number(value,digits=2):
    return '--' if value is None else f'{value:.{digits}f}'

def signed(value,digits=1,scale=1.):
    """Typeset a possibly negative number with a true minus sign."""
    if value is None:return '--'
    text=f'{abs(value*scale):.{digits}f}'
    return '$-$'+text if value*scale<0 and float(text)!=0 else text

def percent(value):
    return signed(value,1,100.)

def interval(bounds):
    return r'\mbox{['+percent(bounds[0])+', '+percent(bounds[1])+']}'

def plural(count,one,many):
    return one if count==1 else many

def words(count):
    return ('zero','one','two','three','four','five','six','seven','eight','nine','ten')[count] if 0<=count<=10 else str(count)

def verdict(n,effect,pv):
    if n==0:
        return 'No valid complete pair was available, so this analysis cannot estimate the region treatment effect.'
    if pv<.05 and effect>0:
        return 'The exact paired test supports a positive effect in this single-task setting; it does not establish broader transfer.'
    if pv<.05 and effect<0:
        return 'The exact paired test indicates worse success for the region presentation in this single-task setting.'
    if effect>0:
        return 'The observed paired difference favors regions, but this small validation does not establish a reliable success advantage.'
    if effect<0:
        return 'The observed paired difference favors point focus; this validation does not support a success benefit from the region presentation.'
    return 'The observed paired success difference is zero; the validation does not establish a region advantage or prove equivalence.'

def summary(arms,p,label):
    n=p['eligible_pairs']
    if not n:return f'{label}, none of the 12 planned pairs met all complete-outcome and admission requirements.'
    return (f'{label}, {n} of 12 pairs are admissible: regions succeeded in {arms["regions"]["paired_eligible_successes"]}/{n} '
            f'and point focus in {arms["point"]["paired_eligible_successes"]}/{n}, with {p["regions_only_success"]} region-only and '
            f'{p["point_only_success"]} point-only {plural(p["point_only_success"],"success","successes")}. '
            f'The paired difference is {percent(p["risk_difference_regions_minus_point"])} percentage points '
            f'(conservative 95\\% interval {interval(p["risk_difference_conservative95"])}, exact paired $p={number(p["mcnemar_exact_two_sided_p"],4)}$).')

def envelope(pairs):
    """Worst-case paired difference over the 12 planned seeds.

    Admitted pairs contribute their observed contrast. For an excluded pair, a
    known arm keeps its outcome and an unknown arm ranges over success/failure;
    a pair excluded with both arms known (failed admission) ranges over [-1, 1].
    """
    lo=hi=0
    for pair in pairs:
        point,regions=pair['point_outcome'],pair['regions_outcome']
        if pair['eligible']:
            delta=int(regions=='success')-int(point=='success');lo+=delta;hi+=delta;continue
        if point!='unknown' and regions!='unknown':
            lo-=1;hi+=1;continue
        ps=[0,1] if point=='unknown' else [int(point=='success')]
        rs=[0,1] if regions=='unknown' else [int(regions=='success')]
        lo+=min(r-q for r in rs for q in ps);hi+=max(r-q for r in rs for q in ps)
    return lo/len(pairs),hi/len(pairs)

def arm_rows(arms,n,label):
    rows=[]
    for arm in ('point','regions'):
        a=arms[arm];iv=a['paired_eligible_success_rate_wilson95']
        paired=a['final_xy_paired_eligible_including_failures'];known=a['final_xy_all_known_including_failures']
        rate=f'{a["paired_eligible_successes"]}/{a["paired_eligible_denominator"]}' if n else '--'
        rows.append(f'{label if arm=="point" else ""} & {arm.capitalize()} & {a["success"]}/{a["task_failure_including_horizon_timeout"]}/{a["unknown"]} & '
                    f'{rate} & {"--" if iv is None else interval(iv)} & {number(paired["median_cm"])} ({paired["n"]}) & {number(known["median_cm"])} ({known["n"]})'+r'\\')
    return rows

def main():
    # The interrupted run is analyzed together with its resume as one 24-case report.
    report_path=Path(sys.argv[1]) if len(sys.argv)>1 else WORK/'final-analysis-v1.json'
    d=json.loads(report_path.read_text(encoding='utf-8'))
    assert d['source_files_match'] is True and len(d['cases'])==24
    manifest=json.loads((WORK/'manifest.json').read_text(encoding='utf-8'))
    assert d['manifest_sha256']==manifest['manifest_sha256']=='9414bd6546022392a2c813f1b8999a050a2c5230592f6f67fcc61971964eb9e4'
    assert d['launch_cleanup']['owned_processes_stopped'] is True
    assert not d['launch_cleanup'].get('cleanup_errors')
    analysis_plan=json.loads((WORK/'analysis-plan-v1.json').read_text(encoding='utf-8'))
    assert d['analysis_sha256']==analysis_plan['source_sha256']['work/id60-regions-validation/analyze_regions_validation.py']
    full_arms,full=d['arms'],d['paired']
    s=d.get('registered_rule_sensitivity')
    # Primary: registered accounting rule. Supplement: the deviation-inclusive analysis.
    arms,p,pairs=(s['arms'],s['paired'],s['pairs']) if s else (full_arms,full,d['pairs'])
    n=p['eligible_pairs'];effect=p['risk_difference_regions_minus_point'];pv=p['mcnemar_exact_two_sided_p']
    by={(c['seed'],c['comparison_arm']):c for c in d['cases']}
    def counts(a):
        known=sum(a[x]['success']+a[x]['task_failure_including_horizon_timeout'] for x in ('point','regions'))
        return known,sum(a[x]['unknown'] for x in ('point','regions'))
    known,unknown=counts(arms)
    conclusion=verdict(n,effect,pv)
    if s:
        # Qualify, so the zero-difference verdict is not read as describing the supplement.
        conclusion='Under the registered rule, '+conclusion[0].lower()+conclusion[1:]
    overall=''
    reattempted=sorted({c['seed'] for c in d['cases'] if c.get('reattempt_after_zero_step_kill')})
    primary=summary(arms,p,'Under the registered accounting rule' if s else 'Across the registered pairs')
    supplement=''
    added=[]
    if s:
        eligible={q['seed'] for q in pairs if q['eligible']}
        added=[q for q in d['pairs'] if q['eligible'] and q['seed'] not in eligible]
        supplement=summary(full_arms,full,'Including the re-executed case, a disclosed post-registration deviation')
        if added:
            extra='; '.join(f'seed {q["seed"]}: point {q["point_outcome"]}, regions {q["regions_outcome"]}' for q in added)
            supplement+=f' The {plural(len(added),"pair","pairs")} added by the deviation ({extra}) {plural(len(added),"accounts","account")} for every difference between the two analyses.'
        effects=[effect,full['risk_difference_regions_minus_point']];pvals=[pv,full['mcnemar_exact_two_sided_p']]
        if all(e<=0 for e in effects):
            overall='Neither analysis supports a success benefit from the region presentation.'
        elif not any(e>0 and v<.05 for e,v in zip(effects,pvals)):
            overall='Neither analysis establishes a success benefit from the region presentation.'
    disruption=''
    unknown_cases=[c for c in d['cases'] if c['outcome']=='unknown']
    if s:
        disruption=('The run was interrupted once by a host shutdown; one case that had recorded no environment step was re-executed after registration'
                    +(f', and {words(len(unknown_cases))} {plural(len(unknown_cases),"case","cases")} ended as an unknown infrastructure outcome' if unknown_cases else '')+'. ')
    # Per-seed ledger.
    ledger=[]
    def steps_of(c):
        if c['steps'] is not None:return str(c['steps'])
        trace=json.loads(Path(c['result_path']).read_text(encoding='utf-8')).get('trace') or {}
        done=trace.get('last_step')
        return f'{done}$^\\ddagger$' if isinstance(done,int) and done>0 else '--'
    for pair in d['pairs']:
        row=[]
        for arm in ('point','regions'):
            c=by[pair['seed'],arm]
            xy=c['final_xy'].get('xy_error_m')
            mark=r'$^\dagger$' if c.get('reattempt_after_zero_step_kill') else ''
            row.extend([{'success':'S','failure':'F','unknown':'U'}[c['outcome']]+mark,steps_of(c),number(xy*100 if xy is not None else None)])
        reg=next((q for q in pairs if q['seed']==pair['seed']),pair)
        row.append(('Yes' if reg['eligible'] else 'No')+('/Yes' if s and pair['eligible'] and not reg['eligible'] else ''))
        ledger.append(' & '.join([str(pair['seed']),*row])+r'\\')
    # Results text.
    # The generated block starts with MARK, so the table float can precede the paragraph that cites it.
    MARK='% Prospective validation: generated by work/id60-regions-validation/update_paper_results.py'
    text=(PAPER/'results.tex').read_text(encoding='utf-8')
    old_results=text.split(MARK)[0] if MARK in text else text.split(r'\paragraph{Prospective validation.}')[0]
    known_full,unknown_full=counts(full_arms)
    new=(MARK+'\n'+r'''\begin{table*}[t]
\centering
\caption{Prospective validation, separate from development. S/F/U are official successes, known task failures, and unknown cases across 12 planned episodes per arm. Paired success, Wilson intervals and the paired distance median use the identical admitted seed pairs of each analysis. Distances are median terminal XY diagnostics in centimeters, including failures, with $n$ in parentheses; the all-known column also includes episodes whose partner is unknown and so compares different seed sets across arms.'''
         +(r' The registered-rule rows are primary; the deviation rows include the re-executed zero-step case (Section~\ref{sec:protocol}).' if s else '')+r'''}
\label{tab:validation}
\small
\begin{tabular}{llrrrrr}\toprule
Analysis & Arm & S/F/U & Paired success & Wilson 95\% (\%) & Median cm, pairs & Median cm, all known\\\midrule
'''+'\n'.join(arm_rows(arms,n,'Registered rule' if s else 'Registered'))
         +(r'\midrule'+'\n'+'\n'.join(arm_rows(full_arms,full['eligible_pairs'],'With deviation')) if s else '')+r'''
\bottomrule\end{tabular}
\end{table*}
'''
         +r'\paragraph{Prospective validation.}'+'\n'
         +disruption
         +f'As executed, the fixed 24-case ledger (Table~\\ref{{tab:ledger}}) contains {known_full} known task outcomes and {unknown_full} unknown {plural(unknown_full,"case","cases")}'
         +(f'; under the registered rule the re-executed case is also unknown ({known} known, {unknown} unknown). ' if s else '. ')
         +primary+f' The matched outcomes comprise {p["both_success"]} shared {plural(p["both_success"],"success","successes")} and {p["both_failure"]} shared {plural(p["both_failure"],"failure","failures")} (Table~\\ref{{tab:validation}}). '
         +conclusion+'\n\n'
         +(r'\paragraph{Including the re-executed case.}'+'\n'+supplement+(' '+overall if overall else '')+'\n\n' if s else ''))
    new+='The conservative effect intervals include zero. They are constructed from simultaneous exact marginal intervals, whereas the conditional paired test uses only discordant outcomes; both are reported without treating them as interchangeable.\n\n'
    xy=p['final_xy_paired_delta_cm']
    if xy['n']:
        new+=(f'For the {xy["n"]} admissible pairs with both terminal distances (Figure~\\ref{{fig:validation}}), the median within-pair regions-minus-point difference was {signed(xy["median"],2)}~cm and the mean was {signed(xy["mean"],2)}~cm. ')
        if s and full['final_xy_paired_delta_cm']['n']:
            fx=full['final_xy_paired_delta_cm']
            new+=f'Including the re-executed case, the corresponding values over {fx["n"]} pairs are {signed(fx["median"],2)}~cm and {signed(fx["mean"],2)}~cm. '
        new+=('Negative values favor regions. This secondary endpoint measures the natural selected site at each episode termination, not a common-time trajectory error or a semantic mouth-to-nozzle constraint. Official success remains the primary outcome.\n\n')
    else:
        new+='No admitted pair has both recorded terminal distances, so the paired distance effect is unavailable and no paired distance figure is shown.\n\n'
    if unknown or unknown_full:
        new+='Unknown cases are retained in the per-seed ledger and are excluded from paired task inference rather than relabeled as failures; complete case accounting does not mean all 24 robot trajectories completed. '
    lo,hi=envelope(pairs)
    new+=(f'Bounding each excluded pair by its known arm and letting each unknown arm be either success or failure gives a worst-case envelope of \\mbox{{[{percent(lo)}, {percent(hi)}]}} percentage points over the 12 planned seeds')
    if s:
        flo,fhi=envelope(d['pairs'])
        new+=f' under the registered rule, and \\mbox{{[{percent(flo)}, {percent(fhi)}]}} including the re-executed case'
    new+='. These envelopes do not include sampling uncertainty and are not confidence intervals.\n'
    if xy['n']:new+=r'''\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{images/validation-pairs.pdf}
\caption{Terminal XY diagnostics for every prospective seed pair with both distances recorded. Each row is one seed; circles denote official success and crosses known failure, and the line joins the two arms of the pair. Seeds with an unknown arm have no row and are not assigned zero error.'''+(r' $\dagger$: the regions case was re-executed after a zero-step host interruption; this pair is excluded under the registered rule.' if reattempted else '')+r'''}
\label{fig:validation}
\end{figure}
'''
    (PAPER/'results.tex').write_text(old_results+new,encoding='utf-8')
    brief=''
    if s and added:
        brief=(f' Including the re-executed case, a disclosed deviation, adds {words(len(added))} {plural(len(added),"pair","pairs")} '
               f'(seed {", ".join(str(q["seed"]) for q in added)}) and gives regions {full_arms["regions"]["paired_eligible_successes"]}/{full["eligible_pairs"]} '
               f'versus point focus {full_arms["point"]["paired_eligible_successes"]}/{full["eligible_pairs"]} '
               f'({percent(full["risk_difference_regions_minus_point"])} percentage points, {interval(full["risk_difference_conservative95"])}, $p={number(full["mcnemar_exact_two_sided_p"],4)}$); '
               f'{plural(len(added),"this pair is","these pairs are")} the only source of that difference.')
    lead=disruption+primary+' '+conclusion+brief+(' '+overall if overall else '')
    abstract=r'''\begin{abstract}
Can finer spatial references help a frozen robot policy place an object more reliably without updating its weights or overriding its native action outputs? We study a simulator-assisted interface that binds stages to persistent entity handles and augments an existing placement marker with a parent-frame reference lattice and an object-center proxy. The original placement destination and native action chunks are preserved. A three-seed development comparison suggested a possible gain but provided only one discordant success pair. We then froze the method and registered 24 episodes on 12 locally unused seeds, counterbalancing arm order and requiring matched initial states and independently generated natural plans. '''+lead+r''' Separately, 74 fixed-frame GPU generations exposed persistent visual verifier errors despite controller and prompt repairs, so the evaluated presentation excludes experimental alignment feedback. The study distinguishes persistent identity, spatial presentation, and reliable physical verification, and explicitly discloses simulator access and the limits of single-task evidence.
\end{abstract}
'''
    (PAPER/'abstract.tex').write_text(abstract,encoding='utf-8')
    (PAPER/'conclusion.tex').write_text(r'\section{Conclusion}'+'\n'+
        'We implemented persistent simulator-backed entity bindings and local spatial references at the input of an unchanged frozen robot policy. '
        +lead+' The result concerns the combined neighborhood, object-proxy and legend treatment; it does not isolate finer symbolic IDs or establish cross-model, cross-benchmark or real-world transfer. '
        'The failed visual-feedback diagnostics further show why structured responses and high confidence cannot substitute for physical verification. The complete registered ledger, source identities and negative evidence are retained for subsequent work.\n',encoding='utf-8')
    details=r'''\section{Prospective Per-Seed Ledger}
\noindent\begin{minipage}{\columnwidth}\settabletype
\centering\small\setlength{\tabcolsep}{2.5pt}
\caption{Complete registered seed ledger. S: success; F: known failure; U: unknown. Steps are executed control steps. XY is the offline terminal diagnostic in centimeters; missing evidence is shown as --. ``Pair'' denotes admission for paired inference'''+(r" under the registered rule (``No/Yes'': admitted only when the re-executed case is included)" if s else '')+r'''. $\dagger$: re-executed once after a host shutdown that preceded its first environment step (Section~\ref{sec:protocol}); unknown under the registered rule. $\ddagger$: steps executed before an infrastructure timeout; no official outcome or terminal diagnostic was recorded.}
\label{tab:ledger}
\begin{tabular}{lrrrrrrc}\toprule
 & \multicolumn{3}{c}{Point} & \multicolumn{3}{c}{Regions} & \\
Seed & Out & Steps & XY & Out & Steps & XY & Pair\\\midrule
'''+'\n'.join(ledger)+r'''
\bottomrule\end{tabular}
\end{minipage}
\paragraph{Statistical interval definitions.}
Per-arm intervals are Wilson 95\% score intervals on the admitted paired subset. The paired risk-difference interval subtracts the endpoints of two 97.5\% Clopper--Pearson intervals, one for region-only success probability and one for point-only success probability. A Bonferroni argument gives at least 95\% simultaneous coverage under the paired Bernoulli sampling model; the resulting difference interval is conservative. The two-sided exact McNemar test conditions on the number of discordant pairs. The preregistered primary contrast is regions minus point.
\paragraph{Frozen identities.}
The validation manifest digest is \texttt{9414bd65...964eb9e4}. The Xiaomi checkpoint asset digest is \texttt{ccaa6517...6fc184}; the Qwen checkpoint asset digest is \texttt{6efa4249...512d6}. These are experiment asset hashes, not upstream repository commit identifiers. Full digests and file mappings are retained in the evidence package.
'''
    for c in unknown_cases:
        reason=c.get('reason') or '; '.join(c['integrity_issues'])
        trace=json.loads(Path(c['result_path']).read_text(encoding='utf-8')).get('trace') or {}
        done=trace.get('last_step')
        if c['category']=='infrastructure_timeout' and 'TimeoutError' in str(reason) and isinstance(done,int):
            body=(f'Unknown outcome (infrastructure timeout). The episode had executed {done} of its 600 steps when one planner/verifier request '
                  f'exceeded the {TIMEOUT_SECONDS}-second client timeout (logged as \\texttt{{{tex(reason)}}}). Following the registered behavior, the worker stopped; '
                  'the case was not repeated, and its pair is excluded from paired inference.')
        else:
            body=tex(c['category'])+'. '+tex(reason)
        details+='\n'+fr'\paragraph{{Seed {c["seed"]}, {c["comparison_arm"]}.}} '+body+'\n'
    (PAPER/'validation-details.tex').write_text(details,encoding='utf-8')
    # Replace prospective wording in the limitations, retaining the scope statements.
    limits=(PAPER/'limitations.tex').read_text(encoding='utf-8').replace('even complete favorable results would not establish','the present results do not establish')
    (PAPER/'limitations.tex').write_text(limits,encoding='utf-8')
    if xy['n'] or (s and full['final_xy_paired_delta_cm']['n']):
        plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
        shown=[q for q in d['pairs'] if all(by[q['seed'],arm]['final_xy'].get('status')=='RECORDED_FINAL_XY' for arm in ('point','regions'))]
        fig,ax=plt.subplots(figsize=(3.4,.28*len(shown)+.8))
        colors={'point':'#C05A36','regions':'#197A8C'}
        for row,q in enumerate(shown):
            values={arm:by[q['seed'],arm]['final_xy']['xy_error_m']*100 for arm in ('point','regions')}
            ax.plot([values['point'],values['regions']],[row,row],color='#9AA3AE',lw=1,zorder=1)
            for k,arm in enumerate(('point','regions')):
                c=by[q['seed'],arm]
                ax.scatter(values[arm],row+(-.12 if arm=='point' else .12),s=26,zorder=3,color=colors[arm],
                           marker='o' if c['outcome']=='success' else 'x',linewidths=1.3)
        ax.set_yticks(range(len(shown)),[str(q['seed'])+(r'$^\dagger$' if q['seed'] in reattempted else '') for q in shown])
        ax.invert_yaxis();ax.set_xlim(left=0)
        ax.set_xlabel('Terminal XY diagnostic (cm)');ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.25)
        # Neutral square handles: the legend maps color to arm only; marker shape encodes outcome.
        handles=[plt.Line2D([],[],marker='s',linestyle='',color=colors[arm],markersize=5,label=arm.capitalize()) for arm in ('point','regions')]
        ax.legend(handles=handles,loc='lower right',fontsize=7,frameon=False,handletextpad=.3)
        fig.tight_layout();fig.savefig(PAPER/'images/validation-pairs.pdf',bbox_inches='tight');fig.savefig(PAPER/'images/validation-pairs.png',dpi=180,bbox_inches='tight');plt.close(fig)
    receipt={'status':'ACTUAL_VALIDATION_RESULTS_INSERTED','analysis_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
             'manifest_sha256':d['manifest_sha256'],'primary_analysis':'registered_rule' if s else 'registered',
             'known_outcomes':known,'unknown':unknown,'eligible_pairs':n,'conclusion':(conclusion+' '+overall).strip(),
             'deviation_inclusive_eligible_pairs':full['eligible_pairs'] if s else None,
             'human_review_required':True,'semantic_claims_not_automatically_accepted':True}
    (WORK/'paper-results-insertion-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
