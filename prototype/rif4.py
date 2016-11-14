import nengo
import nengo.spa as spa
import numpy as np

D = 16
D2 = 64

model = spa.SPA()
with model:
    model.category = spa.State(D)
    
    model.terms = spa.State(D2)
    
    
    def learned(x):
        v = model.get_output_vocab('category')
        cat_animal = v.parse('ANIMAL')
        cat_color = v.parse('COLOR')
        
        sim_animal = cat_animal.dot(x)
        sim_color = cat_color.dot(x)
        
        threshold = 0.4
        v2 = model.get_output_vocab('terms')
        if sim_animal > sim_color and sim_animal > threshold:
            return v2.parse('DOG+CAT+RAT').v
        elif sim_color > sim_animal and sim_color > threshold:
            return v2.parse('RED+BLUE+GREEN').v
        else:
            return v2.parse('0').v
        
    c = nengo.Connection(model.category.all_ensembles[0], model.terms.input, 
                     function=learned, learning_rule_type=nengo.PES())
                     
    
    model.practice = spa.State(D2)
    
    model.error = spa.State(D2)
    
    nengo.Connection(model.terms.output, model.error.input)
    nengo.Connection(model.practice.output, model.error.input, transform=-1)
    
    nengo.Connection(model.error.output, c.learning_rule)
    
    stop_learning = nengo.Node(1)
    for ens in model.error.all_ensembles:
        nengo.Connection(stop_learning, ens.neurons, transform=-1*np.ones((ens.n_neurons, 1)))
        
    just_memorize = nengo.Node(0)
    for ens in model.terms.all_ensembles:
        nengo.Connection(just_memorize, ens.neurons,
                         transform=np.ones((ens.n_neurons, 1))*-5)
        