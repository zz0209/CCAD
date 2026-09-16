"""Export the independent published-explanation result without model inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/'paper'


def main():
    source = ROOT/'artifacts/independent_reuse_20260916/IR01_CONFIRMATION_ANALYSIS.json'
    data = json.loads(source.read_text())
    exported = dict(source_path=source.relative_to(ROOT).as_posix(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), **data)
    (PAPER/'data/infinitive_reuse.json').write_text(json.dumps(exported, indent=2)+'\n')
    groups = [
        [('Geometry', '8 fixed', 'geometry'), ('Calibrated geometry', '8 fixed', 'geometry_gain'), ('Member relation', '8 fixed', 'native')],
        [('Fixed response basis*', '8/input', 'native_response_gain_8'), ('Tangent columns', '8/input', 'native_tangent_relation_8'), ('Finite columns', '8/input', 'native_response_relation_8')],
        [('Fixed response basis*', 'All codes', 'native_response_gain'), ('Tangent columns', 'All codes', 'native_tangent_relation'), ('Finite columns', 'All codes', 'native_response_relation'), ('Request recoding', 'All codes', 'native_reencode')],
        [('Candidate readout', 'Source dirs.', 'raw'), ('Reconstruction readout', 'Source dirs.', 'raw_reconstruction')]]
    text = '\\begin{tabular}{llr}\\toprule\nMethod & Writing & Error\\\\\\midrule\n'
    for gi, rows in enumerate(groups):
        if gi:
            text += '\\midrule\n'
        for label, budget, key in rows:
            text += f"{label} & {budget} & {data['results'][key]['new_request_nrmse']:.3f}\\\\\n"
    text += '\\bottomrule\\end{tabular}\n'
    (PAPER/'tables/infinitive_reuse.tex').write_text(text)
    profile = [r'\begin{tabular}{llrr}\toprule',
               r'& & \multicolumn{2}{c}{Context preceding \emph{to}}\\',
               r'Part deleted & Realization & Predicate & Object\\\midrule']
    rows = []
    for qi, part in [(1, 'Predicate'), (2, 'Object')]:
        for label, method in [('Source', 'source'), ('Target', 'native_response_relation')]:
            values = data['results'][method]['role_profile'][qi]
            rendered = [f'{value:.3f}'.removeprefix('0') for value in values]
            profile.append(f'{part if label == "Source" else ""} & {label} & '+ ' & '.join(rendered)+r'\\')
            rows.append(dict(part=part, method=method, predicate_context=values[0], object_context=values[1]))
    profile.append(r'\bottomrule\end{tabular}')
    (PAPER/'tables/infinitive_role_profile.tex').write_text('\n'.join(profile)+'\n')
    (PAPER/'data/infinitive_role_profile.json').write_text(json.dumps(dict(
        source_path=exported['source_path'], source_sha256=exported['source_sha256'],
        measure='Clean minus intervened next-token log probability of to, averaged across target seeds and contexts within role.',
        rows=rows), indent=2)+'\n')
    print(json.dumps(dict(tables=['paper/tables/infinitive_reuse.tex', 'paper/tables/infinitive_role_profile.tex'], analysis_sha256=exported['source_sha256'])))


if __name__ == '__main__':
    main()
