from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/scientific_reform_20260923/content_transfer'
METHOD_NAMES = {
    'region6': 'Shared region', 'region2': 'Number-only region',
    'region4': 'Gender-only region', 'union6': 'Complete union',
    'union_minus_region6': 'Union without shared',
    'cached64_matched': 'Matched cached credit', 'raw': 'Full-state donor',
}


def main():
    source = ART/'confirmation_v2/RESULTS.json'
    result = json.loads(source.read_text(encoding='utf-8'))
    costs = json.loads((ART/'COST.json').read_text(encoding='utf-8'))
    output = ROOT/'paper/data/frozen_region_content.json'
    output.write_text(json.dumps(dict(result=result, cost=costs, source=str(source)),
                                 indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = [r'\begin{table}[t]\centering', r'\small',
             r'\begin{tabular}{lrrrr}\toprule',
             r'& \multicolumn{2}{c}{TopK} & \multicolumn{2}{c}{Matryoshka}\\',
             r'Frozen operation & $C$ & $I$ & $C$ & $I$\\\midrule']
    for method, label in METHOD_NAMES.items():
        values = []
        for objective in ['topk', 'matryoshka']:
            entry = result['results'][objective]['equal_factor_syntax']['pooled_methods'][method]
            values.extend(entry[key]['estimate'] for key in ['controller', 'distractor'])
        lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in values)+r'\\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{Content transfer through saved target regions. $C$ is the donor-directed change in the predefined answer log odds when the controlling noun changes. $I$ is the absolute change when only the other noun changes. Entries equally average two factors, two syntactic forms and three fixed confirmation targets, retaining all 256 inputs. Larger $C$ and smaller $I$ indicate selective content transfer; neither quantity is an answer-flip rate.}',
        r'\label{tab:frozen_region_content}', r'\end{table}']
    table = ROOT/'paper/tables/frozen_region_content.tex'
    table.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(data=str(output), table=str(table))))


if __name__ == '__main__':
    main()
