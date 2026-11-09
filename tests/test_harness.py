import importlib.util
import json
from pathlib import Path
import random
import sys
import warnings
import zipfile

import pytest

from semeval_task3.cv import Candidate, nested_folds, nested_word_cv
from semeval_task3.data import Dataset, Usage, label_map, load_dataset
from semeval_task3.evaluation import official_score, read_submission, score, word_bootstrap, write_submission

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def official():
    path = ROOT/'work/official-scorer'
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location('official_evaluation', path/'evaluation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_direct(official, gold, pred):
    g = {f'{word};{sid}':v for (_, word, sid),v in gold.items()}
    p = {f'{word};{sid}':v for (_, word, sid),v in pred.items()}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return official.compute_f1_score(g,p,'fixture')


@pytest.mark.parametrize('truth,pred,expected', [
    ([1,0],[1,0],1), ([1,0],[0,1],0), ([1,0],[0,0],0),
    ([0,0],[0,0],0), ([0,0],[1,1],0), ([1,1],[1,1],1),
    ([1,1,0],[1,0,1],.5), ([1,0,0],[1,1,1],.5),
])
def test_binary_edge_cases(official, truth, pred, expected):
    keys = [('SV','w',str(i)) for i in range(len(truth))]
    gold, predictions = dict(zip(keys,truth)),dict(zip(keys,pred))
    assert score(gold,predictions)['average'] == pytest.approx(expected)
    assert official_direct(official,gold,predictions) == pytest.approx(expected)


def test_differential_random_cases(official):
    rng = random.Random(2027)
    for _ in range(200):
        gold, predictions = {}, {}
        for word in range(rng.randint(1,10)):
            for sid in range(rng.randint(1,40)):
                key = 'SV',str(word),str(sid)
                gold[key], predictions[key] = rng.randint(0,1),rng.randint(0,1)
        assert score(gold,predictions)['average'] == pytest.approx(official_direct(official,gold,predictions),abs=1e-12)


def test_word_macro_is_not_usage_micro(official):
    gold = {('SV','big',str(i)):1 for i in range(100)} | {('SV','small','1'):1}
    pred = {k:int(k[1]=='big') for k in gold}
    assert score(gold,pred)['average'] == .5
    assert official_direct(official,gold,pred) == .5


def test_language_macro_and_paired_bootstrap():
    gold = {('SV','a','1'):1,('SV','b','1'):1,('EN','c','1'):1}
    pred = {k:int(k[0]=='SV') for k in gold}
    result = score(gold,pred)
    assert result['average']==.5  # not 2/3
    ci = word_bootstrap(result,draws=100)
    assert ci['lower']==ci['upper']==.5
    zero = word_bootstrap(result,other=result,draws=100)
    assert zero['lower']==zero['upper']==zero['estimate']==0
    perfect = score(gold,gold)
    delta = word_bootstrap(perfect,other=result,draws=100)
    assert delta['lower']==delta['upper']==delta['estimate']==.5
    assert ci == word_bootstrap(result,draws=100)
    with pytest.raises(ValueError,match='identical'):
        word_bootstrap(result,other=score({('SV','a','1'):1},{('SV','a','1'):1}))


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')


@pytest.fixture
def small_data(tmp_path):
    folder=tmp_path/'SV';folder.mkdir()
    write_rows(folder/'definitions.jsonl',[dict(word='skär',definition='pink')])
    write_rows(folder/'usages.jsonl',[dict(word='skär',sentence_id='1',text='En skär bil',pos='JJ',
                                           start=3,end=7,year='1900',period_label='1900-1909')])
    write_rows(folder/'subtask2.jsonl',[dict(word='skär',sentence_id='1',label=1)])
    return tmp_path


def mutate_first(path, **values):
    row=json.loads(path.read_text().splitlines()[0]);row.update(values);write_rows(path,[row])


def test_loader_unicode_no_gold_and_marking(small_data):
    dataset=load_dataset(small_data)
    assert dataset.gold is None
    assert dataset.usages[0].marked_text()=='En <t>skär</t> bil'
    assert dataset.usages[0].year==1900
    assert not hasattr(dataset.usages[0],'label')
    # Inference must not even need the gold file on disk.
    (small_data/'SV/subtask2.jsonl').unlink()
    assert load_dataset(small_data).gold is None


@pytest.mark.parametrize('change', [dict(start=-1),dict(end=100),dict(start=7,end=3),dict(start=True),
                                    dict(text=''),dict(year='unknown'),dict(sentence_id='')])
def test_loader_rejects_broken_input(small_data,change):
    mutate_first(small_data/'SV/usages.jsonl',**change)
    with pytest.raises(ValueError):load_dataset(small_data)


def test_unresolved_span_is_retained_but_not_marked(small_data):
    mutate_first(small_data/'SV/usages.jsonl',start=0,end=3)
    data=load_dataset(small_data,with_gold=True)
    assert len(data.usages)==1 and len(data.gold)==1
    with pytest.raises(ValueError,match='Unresolved'):data.usages[0].marked_text()


@pytest.mark.parametrize('file', ['usages.jsonl','definitions.jsonl','subtask2.jsonl'])
def test_duplicate_dataset_rows(small_data,file):
    path=small_data/'SV'/file;path.write_text(path.read_text()*2)
    with pytest.raises(ValueError,match='Duplicate'):load_dataset(small_data,with_gold=True)


def test_missing_gold_and_definition(small_data):
    mutate_first(small_data/'SV/subtask2.jsonl',sentence_id='unknown')
    with pytest.raises(ValueError,match='coverage'):load_dataset(small_data,with_gold=True)
    mutate_first(small_data/'SV/definitions.jsonl',word='other')
    with pytest.raises(ValueError,match='Missing definition'):load_dataset(small_data)


@pytest.mark.parametrize('label', [True,False,'1',1.0,None,[],2,-1])
def test_invalid_label_types(official,tmp_path,label):
    bad=[dict(word='w',sentence_id='1',label=label)]
    with pytest.raises(ValueError):label_map(bad,'SV')
    gold=tmp_path/'gold.jsonl';pred=tmp_path/'pred.jsonl'
    write_rows(gold,[dict(word='w',sentence_id='1',label=1)]);write_rows(pred,bad)
    with pytest.raises(ValueError):official.evaluate_subtask2(gold,pred)


@pytest.mark.parametrize('bad',[
    [], [dict(word='w',sentence_id='2',label=1)],
    [dict(word='w',sentence_id='1',label=1)]*2,
    [dict(word='w',sentence_id='1')],
])
def test_official_and_local_reject_bad_coverage(official,tmp_path,bad):
    gold=tmp_path/'gold.jsonl';pred=tmp_path/'pred.jsonl';archive=tmp_path/'pred.zip'
    write_rows(gold,[dict(word='w',sentence_id='1',label=1)]);write_rows(pred,bad)
    with pytest.raises(ValueError):official.evaluate_subtask2(gold,pred)
    with zipfile.ZipFile(archive,'w') as z:z.write(pred,'SV_subtask2.jsonl')
    with pytest.raises(ValueError):read_submission(archive,{('SV','w','1')})


@pytest.mark.parametrize('names', [['../SV_subtask2.jsonl'], ['folder/SV_subtask2.jsonl'],
                                  ['SV_subtask2.jsonl','SV_subtask2.jsonl'],['EN_subtask2.jsonl']])
def test_archive_layout_rejected(tmp_path,names):
    path=tmp_path/'bad.zip'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with zipfile.ZipFile(path,'w') as z:
            for name in names:z.writestr(name,'{}\n')
    with pytest.raises(ValueError):read_submission(path,{('SV','w','1')})


def test_official_cli_and_roundtrip(small_data,tmp_path):
    data=load_dataset(small_data,with_gold=True);archive=tmp_path/'pred.zip'
    write_submission(archive,data.gold)
    original=archive.read_bytes();write_submission(archive,data.gold)
    assert archive.read_bytes()==original
    assert read_submission(archive,data.keys)==data.gold
    result=official_score(archive,small_data,ROOT/'work/official-scorer')
    assert result['results']['subtask2']=={'SV':1.0,'average':1.0}


def toy_dataset():
    usages=tuple(Usage('SV',w,str(i),'x','def',0,1,1900,'1900','NN') for w in ('a','b','c','d') for i in range(2))
    return Dataset(usages,{u.key:int(u.sentence_id=='1') for u in usages})


def test_nested_cv_fit_boundaries_and_outer_label_invariance():
    data=toy_dataset();calls=[]
    def fit(train,labels):
        train_keys={u.key for u in train};assert train_keys==set(labels)
        groups={u.group for u in train}
        def predict(test):
            assert not groups & {u.group for u in test}
            calls.append((groups,{u.group for u in test}))
            return {u.key:.8 if u.sentence_id=='1' else .2 for u in test}
        return predict
    candidate=Candidate('fixture',fit,(.1,.5,.9))
    result=nested_word_cv(data,[candidate],bootstrap_draws=100)
    assert result['report']['average']==1
    assert set(result['predictions'])==data.keys
    assert len(calls)==4*(3+1)
    assert all(s['threshold']==.5 for s in result['selections'])
    changed=Dataset(data.usages,{k:1-v if k[1]=='a' else v for k,v in data.gold.items()})
    rerun=nested_word_cv(changed,[candidate],bootstrap_draws=100)
    # Flipping the held-out word's gold labels cannot change its selection or prediction.
    assert result['selections'][0]==rerun['selections'][0]
    assert {k:v for k,v in result['predictions'].items() if k[1]=='a'}=={k:v for k,v in rerun['predictions'].items() if k[1]=='a'}


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True])
def test_cv_rejects_invalid_model_scores(value):
    def fit(train,labels):return lambda test:{u.key:value for u in test}
    with pytest.raises(ValueError,match='finite'):
        nested_word_cv(toy_dataset(),[Candidate('bad',fit,(.5,))],bootstrap_draws=10)


def test_real_data_folds_and_flags():
    data=load_dataset(ROOT/'work/data/dev',with_gold=True)
    assert len(data.usages)==len(data.gold)==1983
    assert sum(bool(u.span_flags) for u in data.usages)==0
    raw=load_dataset(ROOT/'work/data/dev',with_gold=True,repair_spans=False)
    assert sum(bool(u.span_flags) for u in raw.usages)==5
    assert [u.key for u in data.usages]==[u.key for u in raw.usages]  # IDs and order unchanged
    repaired={u.sentence_id:u.text[u.start:u.end] for u in data.usages if u.original_span}
    assert repaired=={'318c8124_131_136':'skära','560a7d4b_0_5':'skärt','562d233c_101_105':'skär',
                      '7ec8d1c8_60_64':'skär','af84b108_21_25':'skär'}
    for u in data.usages:
        if u.original_span:
            assert u.marked_text().count('<t>')==1
    folds=nested_folds(data)
    assert len(folds)==10
    held=[]
    for outer in folds:
        held.append(outer['test_group'])
        assert len(outer['train_groups'])==9 and len(outer['inner'])==9
        assert outer['test_group'] not in outer['train_groups']
        for inner in outer['inner']:
            assert len(inner['train_groups'])==8
            assert inner['validation_group'] not in inner['train_groups']
            assert outer['test_group'] not in inner['train_groups']
            assert outer['test_group'] != inner['validation_group']
    assert len(set(held))==10


def test_span_repair_requires_a_unique_same_length_match():
    from semeval_task3.data import repair_span
    assert repair_span('skär', 'Hela himlen skärt och', 0, 5) == (12, 17)
    assert repair_span('skär', 'skärt och skärt', 4, 9) is None      # two candidates: stay flagged
    assert repair_span('skär', 'inget här alls', 0, 4) is None      # no candidate: stay flagged
    assert repair_span('skär', 'En SKÄR bil', 0, 4) == (3, 7)       # case-insensitive
