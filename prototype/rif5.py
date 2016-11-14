import nengo
import nengo.spa as spa
import numpy as np

D = 256
sD = 32

model = spa.SPA()
with model:
    model.category = spa.State(D, subdimensions=sD)
    
    model.terms = spa.State(D, subdimensions=sD)
    
    
    c = []
    n_sub = len(model.category.state_ensembles.ea_ensembles)
    for i in range(n_sub):
            
        def learned(x):
            v = model.get_output_vocab('category')
            cat_animal = v.parse('ANIMAL').v
            cat_color = v.parse('COLOR').v
            cat_animal = cat_animal[i*sD:(i+1)*sD]
            cat_color = cat_color[i*sD:(i+1)*sD]
            norm_animal = np.linalg.norm(cat_animal)
            norm_color = np.linalg.norm(cat_color)
            
            sim_animal = cat_animal.dot(x)/norm_animal**2
            sim_color = cat_color.dot(x)/norm_color**2
            
            threshold = 0.4
            v2 = model.get_output_vocab('terms')
            if sim_animal > sim_color and sim_animal > threshold:
                return v2.parse('DOG+CAT+RAT').v
            elif sim_color > sim_animal and sim_color > threshold:
                return v2.parse('RED+BLUE+GREEN').v
            else:
                return v2.parse('0').v
        cc = nengo.Connection(model.category.state_ensembles.ea_ensembles[i], model.terms.input, 
                         function=learned, learning_rule_type=nengo.PES(), transform=1)
        c.append(cc)             
    
    model.practice = spa.State(D)
    
    model.error = spa.State(D)
    
    nengo.Connection(model.terms.output, model.error.input)
    nengo.Connection(model.practice.output, model.error.input, transform=-1)
    
    for cc in c:
        nengo.Connection(model.error.output, cc.learning_rule)
    
    stop_learning = nengo.Node(1)
    for ens in model.error.all_ensembles:
        nengo.Connection(stop_learning, ens.neurons, transform=-1*np.ones((ens.n_neurons, 1)))
        
    just_memorize = nengo.Node(0)
    for ens in model.terms.all_ensembles:
        nengo.Connection(just_memorize, ens.neurons,
                         transform=np.ones((ens.n_neurons, 1))*-5)
        