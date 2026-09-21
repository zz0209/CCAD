import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = [('none', 'Unedited'), ('source', 'Source explanation'),
         ('native', 'Fixed member relation'), ('geometry_gain', 'Calibrated geometry'),
         ('raw', 'Code readout'), ('raw_reconstruction', 'Reconstruction readout'),
         ('input_tangent_budget', 'Input-dependent initial'),
         ('input_gain', 'Source-column calibration'), ('input_program', 'Input-dependent trained')]


def identity(path):
    return dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    out = args.receipt or args.directory/'EXPORT.json'
    assert not out.exists(), out
    panel, sources = {}, []
    for family, seeds in [('ADDITIONAL', [3, 4, 5]), ('POOLED', [2, 3, 4, 5])]:
        up = args.directory/f'{family}_UTILITY.json'
        rp = args.directory/f'{family}_RESPONSE.json'
        utility, response = [json.loads(p.read_text()) for p in [up, rp]]
        assert utility['target_seeds'] == response['targets'] == seeds
        assert utility['classifier'] == 'retrained'
        assert utility['bootstrap'] == response['bootstrap'] == 2000
        sources.extend([identity(up), identity(rp)])
        rows = []
        for method, label in NAMES:
            r = utility['results'][method]
            response_value = response['methods'][method]['parts']['mean'] if method in response['methods'] else None
            rows.append(dict(method=method, label=label, response=response_value,
                full_accuracy=r['full']['profession']['mean'],
                part_accuracy=r['parts_mean']['profession']['mean'],
                part_worst_group=r['parts_mean']['worst_group']['mean'],
                pronoun_name=utility['pronouns_minus_names'][method]['mean']['profession']['mean']))
        panel[family] = dict(targets=seeds, rows=rows,
            utility=utility, response=response)

    def table(families, include_full):
        columns = 'lrrrrr' if include_full else 'lrrrr'
        headers = (r'Method & Response error & Full accuracy & Part accuracy & Part WG & Pronoun--name \\'
                   if include_full else r'Method & Response error & Part accuracy & Part WG & Pronoun--name \\')
        lines = [r'\begin{tabular}{'+columns+'}', r'\toprule', headers, r'\midrule']
        for family in families:
            if len(families) > 1:
                label = 'Additional targets 3--5' if family == 'ADDITIONAL' else 'Pooled targets 2--5'
                lines += [r'\multicolumn{'+str(6 if include_full else 5)+r'}{l}{\textit{'+label+r'}} \\']
            for row in panel[family]['rows']:
                values = ['---' if row['response'] is None else f"{row['response']:.3f}"]
                if include_full:
                    values.append(f"{100*row['full_accuracy']:.2f}")
                values += [f"{100*row[k]:.2f}" for k in ['part_accuracy', 'part_worst_group', 'pronoun_name']]
                label = row['label']
                if row['method'] == 'input_program':
                    label = r'\textbf{'+label+'}'
                lines.append(label+' & '+' & '.join(values)+r' \\')
            lines.append(r'\midrule' if family != families[-1] else r'\bottomrule')
        return '\n'.join(lines+[r'\end{tabular}'])+'\n'

    paper = ROOT/'paper'
    paths = [paper/'tables/program_use_main.tex', paper/'tables/program_use_replication.tex',
             paper/'data/program_use_replication.json']
    paths[0].write_text(table(['ADDITIONAL'], True))
    paths[1].write_text(table(['ADDITIONAL', 'POOLED'], True))
    paths[2].write_text(json.dumps(dict(panels=panel, sources=sources,
        scope='Target replication on the existing development cohort. Fixed source explanation, two profession pairs, two confounding directions and one head seed.'), indent=2)+'\n')
    provenance = dict(sources=sources+[identity(args.directory/'DESIGN.json')],
        generator=identity(Path(__file__)), outputs=[identity(p) for p in paths])
    manifest_path = paper/'data/DATA_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['program_use_replication'] = provenance
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    index_path = paper/'EVIDENCE_INDEX.json'
    index = json.loads(index_path.read_text())
    claim_id = 'trained_program_use_target_replication'
    index['claims'] = [c for c in index['claims'] if c['id'] != claim_id]
    index['claims'].append(dict(id=claim_id, paper='Main Section 4; app:program_use_replication',
        result='The same saved input-dependent programs are evaluated for fixed-head fidelity and later classifier use across additional target dictionaries.',
        evidence=provenance['sources']+provenance['outputs'], current_manuscript=True,
        statistics=panel['ADDITIONAL']['utility']['inference']))
    if claim_id not in index['current_manuscript_claim_ids']:
        index['current_manuscript_claim_ids'].append(claim_id)
    index['main_argument_claim_ids'] = ['operation_and_native_equivalence',
        'human_part_prediction_confirmation', 'source_program_dictionary_adaptation',
        'input_dependent_program_confirmation', claim_id]
    locations = dict(
        independent_functional_structure_and_union_use='Appendix app:functional_path',
        external_functional_queries='Appendix app:functional_path',
        functional_response_queries='Appendix app:functional_path',
        arithmetic_equivalent_source_transfer='Appendix app:arithmetic_components',
        role_conditioned_source_functions='Appendix app:arithmetic_components',
        member_resolved_query_prediction='Appendix app:arithmetic_components',
        independent_member_query_confirmation='Appendix app:member_query_confirmation',
        role_defined_member_queries='Appendix app:role_queries',
        functional_part_execution_dependence='Appendix app:arithmetic_components',
        binding_causal_role_realization='Appendix app:binding_protocol',
        reusable_response_member_selection='Appendix app:binding_protocol',
        published_human_explanation_reuse_development='Appendix app:human_reuse',
        paired_full_part_checks='Main Section 3 and Figure 1; app:paired_full_part',
        human_part_prediction_confirmation='Main Section 3; app:human_confirmation',
        source_program_dictionary_adaptation='Main Section 3; app:program_adaptation',
        input_dependent_program_confirmation='Main Sections 2–3 and Figure 2; app:input_program_confirmation',
        trained_program_consumer='Appendix app:input_program_confirmation; original target 2 development study',
        fixed_support_inference_confirmation='Appendix app:input_program_confirmation',
        source_profile_confirmation='Appendix app:source_profile',
        conditional_profile_confirmation='Appendix app:source_profile',
        qwen_field_confirmation='Appendix app:qwen_fields',
        independent_grammatical_explanation_reuse='Main Section 3; app:grammar_explanation',
        human_explanation_new_task_confirmation='Main Section 4; app:new_task_reuse',
        human_conditional_restoration_confirmation='Appendix app:human_restoration')
    for claim in index['claims']:
        if claim['id'] in locations:
            claim['paper'] = locations[claim['id']]
        claim['main_argument'] = claim['id'] in index['main_argument_claim_ids']
        for record in claim.get('evidence', []):
            path = Path(record.get('path', ''))
            if path.as_posix().startswith('paper/sections/') and (ROOT/path).is_file():
                record['bytes'] = (ROOT/path).stat().st_size
                record['sha256'] = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    index['manuscript'] = dict(entry='paper/main.pdf', editable_source='paper/main.tex',
        identity_receipt='paper/build/BUILD_RECEIPT.json',
        main_method='Input-dependent source-member columns trained through the complete intervention program.',
        current_round_report='artifacts/reuse_generalization_20260921_round07/REPORT.md',
        replication_scope=panel['ADDITIONAL']['utility']['inference'],
        previous_index='artifacts/reuse_generalization_20260921_round07/pre_integration/EVIDENCE_INDEX.json')
    index['manifest_identities'] = dict(data_manifest=identity(manifest_path),
        figure_manifest=identity(paper/'figures/FIGURE_MANIFEST.json'))
    index['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    index_path.write_text(json.dumps(index, indent=2)+'\n')
    out.write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(outputs=[p.as_posix() for p in paths])))


if __name__ == '__main__':
    main()
