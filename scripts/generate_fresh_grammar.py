"""Draw new BLiMP grammar samples using the pinned authors' generator classes.

The original source is retained byte-for-byte. Runtime adaptations only fix
Windows vocabulary lookup and omit the unused jsonlines writer import. We
call sample(), not the upstream unbounded generation/write loop.
"""
import argparse, ast, hashlib, importlib, json, random, sys, types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--old-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=290913)
    p.add_argument('--count',type=int,default=144)
    p.add_argument('--tasks',nargs='+',default=['regular_plural_subject_verb_agreement_1','anaphor_number_agreement','anaphor_gender_agreement'])
    p.add_argument('--split-rule',default='First16 accepted pairs per task are calibration; next128 evaluation. No model outcomes used for generation or exclusion.')
    a=p.parse_args();root=a.source.resolve();a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'DATA_MANIFEST.json').exists(), 'Do not overwrite a generated panel'
    import numpy as np
    random.seed(a.seed);np.random.seed(a.seed)
    sys.path.insert(0,str(root));importlib.import_module('utils')
    for name in ['utils.data_type','utils.vocab_table','utils.data_generator']:
        path=root/(name.replace('.','/')+'.py');tree=ast.parse(path.read_text())
        if name.endswith('data_type'):
            # The upstream100000-character expression buffer makes small lexical
            # arrays consume many GB. These three nonrecursive grammars produce
            # short clauses;4096 retains a large margin over every observed value.
            tree=ast.parse(path.read_text().replace('"U100000"','"U4096"'))
        elif name.endswith('vocab_table'):
            for i,node in enumerate(tree.body):
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='vocab_path' for t in node.targets):
                    tree.body[i]=ast.parse('vocab_path = '+repr(str(root/'vocabulary.csv'))).body[0]
        else:
            tree.body=[n for n in tree.body if not (isinstance(n,ast.Import) and any(x.name=='jsonlines' for x in n.names))]
        ast.fix_missing_locations(tree);module=types.ModuleType(name);module.__file__=str(path)
        sys.modules[name]=module;exec(compile(tree,str(path),'exec'),module.__dict__)
    from utils.string_utils import string_beautify
    tasks=a.tasks
    old=json.loads(a.old_manifest.read_text());seen=set();oldfiles=[]
    for fi in old['files']:
        fp=Path(fi['path']);assert digest(fp)==fi['sha256'];oldfiles.append(fi)
        for line in fp.read_text().splitlines():
            row=json.loads(line)
            # Exclude even one previously seen good/bad sentence, not just exact pairs.
            seen.update([row['sentence_good'],row['sentence_bad']])
    files=[];allcounts={};examples={}
    for task in tasks:
        src=root/'generation_projects/blimp'/(task+'.py');tree=ast.parse(src.read_text())
        # The reviewed modules contain imports, one class, and an automatic writer.
        tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef))]
        ns={'__file__':str(src)};exec(compile(tree,str(src),'exec'),ns)
        classes=[n.name for n in tree.body if isinstance(n,ast.ClassDef)]
        assert len(classes)==1,(task,classes)
        gen=ns[classes[0]]()
        rows=[];counts=Counter()
        for attempt in range(a.count*200):
            try:data,_=gen.sample()
            except (IndexError,ValueError,np.exceptions.DTypePromotionError) as e:
                counts[type(e).__name__]+=1;continue
            for field in gen.data_fields:
                if field in data:data[field]=string_beautify(data[field])
            good,bad=data['sentence_good'],data['sentence_bad']
            assert max(len(good),len(bad))<1024, 'Unexpected long clause: inspect expression-buffer adaptation'
            if good in seen or bad in seen:counts['previous_sentence_or_duplicate']+=1;continue
            if good==bad:counts['identical_good_bad_draw']+=1;continue
            assert good.startswith(data['one_prefix_prefix']) and bad.startswith(data['one_prefix_prefix']),(task,data)
            data.update(gen.make_metadata_dict(),pairID=str(len(rows)),generation_seed=a.seed)
            seen.update([good,bad]);rows.append(data)
            if len(rows)==a.count:break
        assert len(rows)==a.count,(task,len(rows),counts)
        path=a.out/(task+'.jsonl');content=''.join(json.dumps(r)+'\n' for r in rows)
        if path.exists():assert path.read_text()==content, 'Existing partial sample changed under replay'
        else:path.write_text(content,encoding='utf-8')
        files.append(dict(task=task,path=str(path.resolve()),sha256=digest(path),bytes=path.stat().st_size,used=True))
        allcounts[task]=dict(accepted=len(rows),**counts);examples[task]=rows[:3]
        print(task,allcounts[task],flush=True)
    manifest=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),files=files,seed=a.seed,
        generator_commit='7b93dccc9a773d76cfdbc5391f7087e34eefd3cf',generator_manifest_sha256=digest(root/'DOWNLOAD_MANIFEST.json'),
        wrapper_sha256=digest(__file__),excluded_original_files=oldfiles,counts=allcounts,
        split_rule=a.split_rule,
        exclusion_scope='Every good/bad sentence from every file in the supplied manifest, including previous generated panels when present.',
        adaptations=['Windows vocabulary path supplied explicitly','Expression Unicode buffer reduced100000 to4096 characters for three nonrecursive short-clause grammars; every accepted sentence must be shorter than1024 characters','Unused jsonlines import omitted; upstream sample/classes/beautification unchanged','Top-level automatic generation omitted; bounded sampling and exact-sentence exclusion performed by this wrapper'],
        provenance='New samples from the authors BLiMP grammar generators and vocabulary, not the original1000-pair benchmark or newly human-validated labels. Same grammar/lexicon distribution; disjoint sentences from all locally retained original BLiMP files.',
        license='Official BLiMP dataset is CC BY4.0. Generator README invites use/citation but repository has no formal code LICENSE. Third-party source retained locally, not included in project code redistribution.')
    (a.out/'DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.out/'EXAMPLES.json').write_text(json.dumps(examples,indent=2)+'\n')


if __name__=='__main__':main()
