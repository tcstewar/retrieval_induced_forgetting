import numpy as np
import pylab
import nengo
import nengo.spa as spa
import random as rnd
import scipy.stats

class RIFModelPhase1(object):
    def __init__(self, mapping, D_category=16, D_items=64, threshold=0.4, learning_rate=1e-4):
        model = spa.SPA()
        self.model = model
        self.mapping = mapping
        self.vocab_category = spa.Vocabulary(D_category)
        self.vocab_items = spa.Vocabulary(D_items)
        for k in sorted(mapping.keys()): #allocating verctors for categories name
            self.vocab_category.parse(k)
            for v in mapping[k]: # allocating vectors to the items
                self.vocab_items.parse(v)
        
        with model:
            model.category = spa.State(D_category, vocab=self.vocab_category)
        
            model.items = spa.State(D_items, vocab=self.vocab_items)

            def learned(x):
                #cats = np.dot(self.vocab_category.vectors, x)
                #best_index = np.argmax(cats) #takes the category which has the largest projection of x
                #if cats[best_index] < threshold:
                #   return self.vocab_items.parse('0').v
                #else: #generate the sum vector
                #    k = self.vocab_category.keys[best_index]
                #    total = '+'.join(self.mapping[k])
                #    v = self.vocab_items.parse(total).v
                return self.vocab_items.parse('0').v# v/(2*np.linalg.norm(v))
                
            c = nengo.Connection(model.category.all_ensembles[0], model.items.input, 
                             function=learned, learning_rule_type=nengo.PES(learning_rate=learning_rate))

            
            model.error = spa.State(D_items, vocab=self.vocab_items)
            nengo.Connection(model.items.output, model.error.input)
            nengo.Connection(model.error.output, c.learning_rule)
            
            
            self.stim_category_value = np.zeros(D_category)
            self.stim_category = nengo.Node(self.stim_category)
            nengo.Connection(self.stim_category, model.category.input, synapse=None)

            self.stim_correct_value = np.zeros(D_items)
            self.stim_correct = nengo.Node(self.stim_correct)
            nengo.Connection(self.stim_correct, model.error.input, synapse=None, transform=-1)

            self.stim_stoplearn_value = np.zeros(1)
            self.stim_stoplearn = nengo.Node(self.stim_stoplearn)
            for ens in model.error.all_ensembles:
                nengo.Connection(self.stim_stoplearn, ens.neurons, synapse=None, transform=-10*np.ones((ens.n_neurons, 1)))

            self.stim_justmemorize_value = np.zeros(1)
            self.stim_justmemorize = nengo.Node(self.stim_justmemorize)
            for ens in model.items.all_ensembles:
                nengo.Connection(self.stim_justmemorize, ens.neurons, synapse=None, transform=-10*np.ones((ens.n_neurons, 1)))
         
            
            
            self.probe_items = nengo.Probe(model.items.output, synapse=0.01)
            
        self.sim = nengo.Simulator(self.model)
        
    def stim_category(self, t):
        return self.stim_category_value

    def stim_correct(self, t):
        return self.stim_correct_value
    
    def stim_stoplearn(self, t):
        return self.stim_stoplearn_value

    def stim_justmemorize(self, t):
        return self.stim_justmemorize_value
   
    def test(self, category,items=None, T=0.5):
        self.stim_justmemorize_value = 0 #Terry's paranoia
        self.stim_stoplearn_value = 1
        self.stim_category_value = self.vocab_category.parse(category).v
        self.stim_correct_value = self.vocab_items.parse('0').v
        self.sim.run(T)
        d = self.sim.data[self.probe_items]
        index=[]
        if items is None:
            items=self.mapping[category]
        for item in items:
            index.append(self.vocab_items.keys.index(item))
        
        #self.sim.data[self.probe_items] - this should stay commented
        return np.dot(self.vocab_items.vectors, d[-1]) [index]
        
    def practice(self, category, item, T=0.5):
        self.stim_justmemorize_value = 0 #Terry's paranoia
        self.stim_stoplearn_value = 0
        self.stim_category_value = self.vocab_category.parse(category).v
        self.stim_correct_value = self.vocab_items.parse(item).v
        self.sim.run(T)

    def practice_reverse(self, category,item,cue,T=0.5):
        self.stim_justmemorize_value = 0 #Terry's paranoia
        self.stim_stoplearn_value = 0
        if cue=='category':
            self.stim_category_value = self.vocab_category.parse(category).v
            self.stim_correct_value = self.vocab_items.parse(item).v
        else:
            self.stim_correct_value = self.vocab_category.parse(category).v
            self.stim_category_value = self.vocab_items.parse(item).v
            
        self.sim.run(T)


    def memorize(self, category, item, T=0.5): #for the simulation without retri
        self.stim_stoplearn_value = 0
        self.stim_justmemorize_value = 1
        self.stim_category_value = self.vocab_category.parse(category).v
        self.stim_correct_value = self.vocab_items.parse(item).v
        self.sim.run(T)
        self.stim_justmemorize_value = 0

        

 
