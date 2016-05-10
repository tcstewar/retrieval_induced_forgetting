import nengo
import nengo.spa as spa
import ctn_benchmark
import numpy as np

class RIF(ctn_benchmark.Benchmark):
    def params(self):
        self.default('dimensions', D=32)
        self.default('consolidation strength', strength=1.0)
    def model(self, p):
        model = spa.SPA()
        with model:
            model.memory = spa.State(p.D, feedback=1)
            model.cue = spa.State(p.D)

            model.input = spa.Input(memory=lambda t: 'P1*DOG+P2*CAT+P3*RAT' if t<0.1 else '0',
                                    cue='P3')



            model.item = spa.State(p.D)


            model.cortical = spa.Cortical(
                spa.Actions(
                    'item = memory*~cue',
                    'memory = %g*item*cue' % p.strength,
                    ))

            self.p_memory = nengo.Probe(model.memory.output, synapse=0.03)
        self.vocab = model.get_output_vocab('memory')
        return model

    def evaluate(self, p, sim, plt):
        sim.run(0.5)

        if plt is not None:
            data = sim.data[self.p_memory]
            for term in ['P1*DOG', 'P2*CAT', 'P3*RAT']:
                plt.plot(sim.trange(), np.dot(data, self.vocab.parse(term).v), label=term)
            plt.legend(loc='upper left')

        return {}

if __name__ == '__main__':
    RIF().run()





