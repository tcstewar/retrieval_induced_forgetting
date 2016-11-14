import numpy as np
import pylab
import nengo
import nengo.spa as spa
import random as rnd
import scipy.stats

sD=16
threshold=0.4

class RIFModel(object):
    def __init__(self, mapping, threshold=0.4, learning_rate=1e-4,DimVocab=256):
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
            #model.category = spa.State(D_category, vocab=self.vocab_category)
            #model.items = spa.State(D_items, vocab=self.vocab_items)
            model.cue = spa.State(DimVocab, vocab=self.VocabUnified)
            model.target = spa.State(DimVocab, vocab=self.VocabUnified)
            c = []
            n_sub = len(model.cue.state_ensembles.ea_ensembles)
            for i in range(n_sub):
                def learned(x):
                    cue = np.dot(self.VocabUnified.vectors[:,i*sD:(i+1)*sD],x)
                    best_index = np.argmax(cue) #takes the category which has the largest projection of x
                    specific_cue=cue[best_index]
                    norm_specific_cue=np.linalg.norm(specific_cue)
                    specific_cue=specific_cue/(norm_specific_cue**2)
                    if specific_cue < threshold:
                        return self.VocabUnified.parse('0').v
                    else: #generate the sum vector
                        k = self.VocabUnified.keys[best_index]
                        if k in mapping.keys():
                            total = '+'.join(self.mapping[k])
                        else:
                            for key in mapping:
                                if k in mapping[key]:
                                    total = key
                                    
                        v = self.VocabUnified.parse(total).v
                        return v/((2*np.linalg.norm(v))*n_sub)                  
                cc = nengo.Connection(model.cue.all_ensembles[i], model.target.input, 
                             function=learned, learning_rule_type=nengo.PES(learning_rate=learning_rate))
                c.append(cc)
                print i
                
            #model.error = spa.State(D_items, vocab=self.vocab_items)
            #nengo.Connection(model.items.output, model.error.input)
            #nengo.Connection(model.error.output, c.learning_rule)

            #I am not sure how this implements the learning
            model.error = spa.State(DimVocab, vocab=self.VocabUnified)
            nengo.Connection(model.target.output, model.error.input)
            print 'the loop ended, right?'
            for cc in c:
                nengo.Connection(model.error.output, cc.learning_rule)
            
            self.cue_value = np.zeros(DimVocab) #?
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
            
        self.sim = nengo.Simulator(self.model)
        
    def stim_cue_fun(self, t):
        return self.cue_value

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
        
        #self.sim.data[self.probe_items] - this should stay commented
        return np.dot(self.VocabUnified.vectors, d[-1]) [index]
        
    #def practice(self, cue, target, T=0.5):
    #    self.stim_justmemorize_value = 0 #Terry's paranoia
    #    self.stim_stoplearn_value = 0
    #    self.stim_cue_value = self.VocabUnified.parse(cue).v
    #    self.stim_correct_value = self.VocabUnified.parse(target).v
    #    self.sim.run(T)

    def practice(self,cue,target,cond,T=0.5):
        self.stim_justmemorize_value = 0 #Terry's paranoia
        self.stim_stoplearn_value = 0
        if cond=='category':
            self.stim_cue_value = self.VocabUnified.parse(cue).v
            self.stim_correct_value = self.VocabUnified.parse(target).v
        else: 
            self.stim_target_value = self.VocabUnified.parse(target).v
            self.stim_correct_value = self.VocabUnified.parse(cue).v
            
        self.sim.run(T)


    def memorize(self, cue, target, T=0.5): #for the simulation without retri
        self.stim_stoplearn_value = 0
        self.stim_justmemorize_value = 1
        self.stim_cue = self.VocabUnified.parse(cue).v 
        self.stim_correct_value = self.VocabUnified.parse(target).v
        self.sim.run(T)
        self.stim_justmemorize_value = 0
