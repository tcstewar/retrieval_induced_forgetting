import numpy as np
import pylab
import nengo
import nengo.spa as spa
import random as rnd
import scipy.stats

class RIFModelRev(object):
    def __init__(self, mapping, threshold=0.2, learning_rate=1e-4,DimVocab=256, subdimensions=32,
                 learned_function_scale=2.0):
        model = spa.SPA()
        self.model = model
        self.mapping = mapping
        #self.vocab_category = spa.Vocabulary(D_category)
        #self.vocab_items = spa.Vocabulary(D_items)
        self.VocabUnified=spa.Vocabulary(DimVocab) 
        for k in sorted(mapping.keys()): #allocating verctors for categories name
            self.VocabUnified.parse(k)
            for v in mapping[k]: # allocating vectors to the items
                self.VocabUnified.parse(v)
        
        with model:
            model.cue = spa.State(DimVocab, vocab=self.VocabUnified, subdimensions=subdimensions)
            model.target = spa.State(DimVocab, vocab=self.VocabUnified, subdimensions=subdimensions)
            c = []
            n_sub = len(model.cue.state_ensembles.ea_ensembles)
            for i in range(n_sub):
                cues = []
                targets = []
                for cue, vals in mapping.items():
                    for val in vals:
                        cues.append(self.VocabUnified.parse(cue).v[i*subdimensions:(i+1)*subdimensions])
                        targets.append(self.VocabUnified.parse(val).v/n_sub*learned_function_scale)
                        cues.append(self.VocabUnified.parse(val).v[i*subdimensions:(i+1)*subdimensions])
                        targets.append(self.VocabUnified.parse(cue).v/n_sub*learned_function_scale)

                cc = nengo.Connection(model.cue.all_ensembles[i], model.target.input, 
                             learning_rule_type=nengo.PES(learning_rate=learning_rate),
                             **nengo.utils.connection.target_function(cues, targets))
                cc.eval_points=cues
                cc.function = targets
                c.append(cc)

                
                print i
                
            #model.error = spa.State(D_items, vocab=self.vocab_items)
            #nengo.Connection(model.items.output, model.error.input)
            #nengo.Connection(model.error.output, c.learning_rule)

            #I am not sure how this implements the learning
            model.error = spa.State(DimVocab, vocab=self.VocabUnified, subdimensions=subdimensions)
            nengo.Connection(model.target.output, model.error.input)
            print 'the loop ended, right?'
            for cc in c:
                nengo.Connection(model.error.output, cc.learning_rule)
            
            self.stim_cue_value = np.zeros(DimVocab) #?
            self.stim_cue = nengo.Node(self.stim_cue_fun)#?
            nengo.Connection(self.stim_cue, model.cue.input, synapse=None)

            self.stim_correct_value = np.zeros(DimVocab)
            self.stim_correct = nengo.Node(self.stim_correct)
            nengo.Connection(self.stim_correct, model.error.input, synapse=None, transform=-1)

            self.stim_stoplearn_value = np.zeros(1)
            self.stim_stoplearn = nengo.Node(self.stim_stoplearn)
            for ens in model.error.all_ensembles:
                nengo.Connection(self.stim_stoplearn, ens.neurons, synapse=None, transform=-10*np.ones((ens.n_neurons, 1)))

            self.stim_justmemorize_value = np.zeros(1)
            self.stim_justmemorize = nengo.Node(self.stim_justmemorize)
            for ens in model.target.all_ensembles: #?
                nengo.Connection(self.stim_justmemorize, ens.neurons, synapse=None, transform=-10*np.ones((ens.n_neurons, 1)))
         
            
            
            self.probe_target = nengo.Probe(model.target.output, synapse=0.01)
        
        if __name__ != '__builtin__':   
            self.sim = nengo.Simulator(self.model)
        
    def stim_cue_fun(self, t):
        return self.stim_cue_value

    def stim_correct(self, t):
        return self.stim_correct_value
    
    def stim_stoplearn(self, t):
        return self.stim_stoplearn_value

    def stim_justmemorize(self, t):
        return self.stim_justmemorize_value
   
    def test(self, cue,items=None, T=0.5):
        self.stim_justmemorize_value = 0 #Terry's paranoia
        self.stim_stoplearn_value = 1
        self.stim_cue_value = self.VocabUnified.parse(cue).v
        self.stim_correct_value = self.VocabUnified.parse('0').v
        self.sim.run(T)
        d = self.sim.data[self.probe_target]
        index=[]
        if items is None:
            items=self.mapping[cue]
        for item in items:
            index.append(self.VocabUnified.keys.index(item))
        
        return np.dot(self.VocabUnified.vectors, d[-1]) [index]
        
    def practice(self, cue, target, T=0.5):
        self.stim_justmemorize_value = 0 #Terry's paranoia
        self.stim_stoplearn_value = 0
        self.stim_cue_value = self.VocabUnified.parse(cue).v
        self.stim_correct_value = self.VocabUnified.parse(target).v
        self.sim.run(T)

    def practice_reverse(self,cue,target,cond,T=0.5):
        self.stim_justmemorize_value = 0 #Terry's paranoia
        self.stim_stoplearn_value = 0
        if cond=='category':
            self.stim_cue_value = self.VocabUnified.parse(cue).v
            self.stim_correct_value = self.VocabUnified.parse(target).v
        else: 
            self.stim_cue_value = self.VocabUnified.parse(target).v
            self.stim_correct_value = self.VocabUnified.parse(cue).v
            
        self.sim.run(T)


    def memorize(self, cue, target, T=0.5): #for the simulation without retri
        self.stim_stoplearn_value = 0
        self.stim_justmemorize_value = 1
        self.stim_cue = self.VocabUnified.parse(cue).v 
        self.stim_correct_value = self.VocabUnified.parse(target).v
        self.sim.run(T)
        self.stim_justmemorize_value = 0
        
if __name__ == '__builtin__':
    mapping = {
    'DRINKS': ['VODKA', 'BOURBON', 'RUM','ALE','GIN','WHISKEY'],
    'WEAPONS': ['SWORD','RIFLE','TANK','BOMB','PISTOL','CLUB'],
    'FISH': ['CATFISH','HERRING','TROUT','BLUGILL','FLOUNDER','GUPPY'],
    'FRUITS' : ['TOMATO','STRAWBERRY','BANANA','ORANGE','LEMON','PINEAPPLE'],
    'PROFESSIONS' : ['ENGINEER','ACCOUNTANT','DENTIST','NURSE','PLUMBER','FARMER'],
    'METALS' : ['IRON','ALUMINUM','NICKEL','SILVER','BRASS','GOLD'],
    'TREES' : ['BIRCH','HICKORY','DOGWOOD','ELM','SPRUCE','REDWOOD'],
    'INSECTS' : ['BEETLE','ROACH','HORNET','FLY','MOSQUITO','GRASSHOPPER']    
    }        
    m = RIFModelRev(mapping, learning_rate=1e-5,DimVocab=256, subdimensions=32, threshold=0.2)
    model = m.model
    m.stim_stoplearn_value = 1
