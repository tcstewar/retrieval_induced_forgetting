import nengo
import nengo.spa as spa

D = 128

model = spa.SPA()
with model:
    model.memory = spa.State(D, feedback=1)
    
    model.input = spa.Input(memory=lambda t: 'P1*DOG+P2*CAT+P3*RAT' if t<0.5 else '0')
    
    
    model.cue = spa.State(D)
    
    model.item = spa.State(D)
    
    
    model.cortical = spa.Cortical(
        spa.Actions(
            'item = memory*~cue',
            'memory = 0.5*item*cue',
            ))
            
    
            
    
    