from contextlib import contextmanager
from utils.experiment import run_epochs


def test_selection_and_checkpoint_run_with_validation_context():
    state={'active':False, 'saved':[]}
    @contextmanager
    def context():
        state['active']=True
        try: yield
        finally: state['active']=False
    def capture(epoch):
        assert state['active']
        return {'epoch':epoch+1}
    def save(epoch,best,history,checkpoint):
        assert not state['active']
        state['saved'].append((best,checkpoint['epoch']))
    best,history,checkpoint=run_epochs(0,3,-1,[],None,
        train_epoch=lambda e:e,validate=lambda e:{'accuracy':[.5,.8,.7][e]},
        capture=capture,record=lambda e,tr,val:{'epoch':e+1,'score':val['accuracy']},
        save_epoch=save,validation_context=context)
    assert best==.8 and checkpoint['epoch']==2
    assert state['saved']==[(.5,1),(.8,2),(.8,2)]
    assert len(history)==3


def test_official_protocol_can_replace_on_equal_validation_score():
    _,_,checkpoint=run_epochs(0,2,-1,[],None,train_epoch=lambda e:None,
        validate=lambda e:{'accuracy':.9},capture=lambda e:{'epoch':e+1},
        record=lambda e,tr,val:{'epoch':e+1},save_epoch=lambda *args:None,replace_on_tie=True)
    assert checkpoint['epoch']==2
