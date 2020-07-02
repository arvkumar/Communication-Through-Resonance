import nest
import time
import numpy
import numpy as np

import matplotlib as mpl

import nest.raster_plot
import pylab
import matplotlib.pyplot as plt

nest.ResetKernel()
nest.SetKernelStatus({'print_time': True,'local_num_threads':12})

msd = 497

n_vp=nest.GetKernelStatus('total_num_virtual_procs')
msdrange1=range(msd,msd+n_vp)

pyrngs=[np.random.RandomState(s) for s in msdrange1]
msdrange2=range(msd+n_vp+1,msd+1+2*n_vp)
nest.SetKernelStatus({'grng_seed':msd+n_vp , 'rng_seeds':msdrange2})

a = 25                 # number of spikes in one pulse packet
sdev = 2.0             # width of pulse packet (ms)
weight = 0.10          # PP amplitude (mV)
pulse_times = [800.]   # occurrence time (center) of pulse-packet (ms)

NP        = 70
NE        = 200             # number of excitatory neurons
NI        = 50              # number of inhibitory neurons
N_neurons = NE+NI           # number of neurons in total
N_rec     = 10*N_neurons    # record from all neuron

simtime    = 1700.     # Simulation time in ms
in_delay   = 1.5       # within-layer synaptic delay in ms

p_rate_ex = 8000.0

bet_g      = 0.32       # connection strength between layers
bet_delay  = 12.5       # between-layer synaptic delay in ms
bet_g_reso      = bet_g           # connection strength between layers
bet_delay_reso  = bet_delay       # between-layer synaptic delay in ms

p_rate_in = p_rate_ex - 1600.0
j_exc_exc = 0.33      # EE connection strength
j_exc_inh = 1.5        # EI connection strength
j_inh_exc = -6.2       # IE connection strength
j_inh_inh = -12.0       # II connection strength


epsilonEE   = 0.2       # EE connection probability
epsilonIE   = 0.2       # IE connection probability
epsilonEI   = 0.2       # EI connection probability
epsilonII   = 0.2       # II connection probability
epsilonP    = 0.2       # inter-layer connection probability

CEE   = int(epsilonEE*NE)      # number of excitatory synapses per E neuron
CIE   = int(epsilonIE*NI)      # number of inhibitory synapses per E neuron
CEI   = int(epsilonEI*NE)      # number of excitatory synapses per I neuron
CII   = int(epsilonII*NI)      # number of inhibitory synapses per I neuron 

CEP    = int(epsilonP*NE)    # number of pulse packet connections per E neuron 
  
exci_neuron_params= {'V_th' :-54.,
                'V_reset'   :-70.,
                'tau_syn_ex': 1.0,
                'tau_syn_in': 1.,
                'tau_minus' : 20.}

inhi_neuron_params= {'V_th' :-54.,
                'V_reset'   :-70.,
                'tau_syn_ex': 1.0,
                'tau_syn_in': 1.,
                'tau_minus' : 20.}

nest.SetKernelStatus({"overwrite_files": True})

pop1_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop1_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)

pop2_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop2_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)

pop3_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop3_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)

pop4_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop4_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)

pop5_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop5_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)

pop6_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop6_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)

pop7_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop7_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)

pop8_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop8_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)

pop9_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop9_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)

pop10_nodes_ex = nest.Create("iaf_cond_alpha",NE,params=exci_neuron_params)
pop10_nodes_in = nest.Create("iaf_cond_alpha",NI,params=inhi_neuron_params)

nodes= pop1_nodes_ex + pop1_nodes_in + pop2_nodes_ex + pop2_nodes_in + pop3_nodes_ex + pop3_nodes_in + pop4_nodes_ex + pop4_nodes_in + pop5_nodes_ex + pop5_nodes_in + pop6_nodes_ex + pop6_nodes_in + pop7_nodes_ex + pop7_nodes_in + pop8_nodes_ex + pop8_nodes_in + pop9_nodes_ex + pop9_nodes_in + pop10_nodes_ex + pop10_nodes_in

nodes_ex= pop1_nodes_ex + pop2_nodes_ex + pop3_nodes_ex + pop4_nodes_ex + pop5_nodes_ex + pop6_nodes_ex + pop7_nodes_ex + pop8_nodes_ex + pop9_nodes_ex + pop10_nodes_ex

nodes_in= pop1_nodes_in + pop2_nodes_in + pop3_nodes_in + pop4_nodes_in + pop5_nodes_in + pop6_nodes_in + pop7_nodes_in + pop8_nodes_in + pop9_nodes_in + pop10_nodes_in 

node_info=nest.GetStatus(nodes)
local_nodes=[(ni['global_id'],ni['vp']) 
             for ni in node_info if ni ['local']]
for gid,vp in local_nodes:
  nest.SetStatus([gid],{'V_m':pyrngs[vp].uniform(-75.0,-65.0)})


pp = nest.Create('pulsepacket_generator',params={'activity':a,'sdev':sdev,'pulse_times':pulse_times})

noise = nest.Create("poisson_generator", 2, [{"rate": p_rate_ex}, {"rate": p_rate_in}])

nest.CopyModel("static_synapse","EI",{"weight":j_exc_inh,"delay":in_delay})
nest.CopyModel("static_synapse","EE",{"weight":j_exc_exc,"delay":in_delay})
nest.CopyModel("static_synapse","IE",{"weight":j_inh_exc,"delay":in_delay})
nest.CopyModel("static_synapse","II",{"weight":j_inh_inh,"delay":in_delay})


##### POPULATION 1
nest.Connect(pop1_nodes_ex,pop1_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE, 'autapses': False},'EE')

nest.Connect(pop1_nodes_ex,pop1_nodes_in,{'rule': 'fixed_indegree','indegree': CEI, 'autapses': False},'EI')

nest.Connect(pop1_nodes_in,pop1_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE, 'autapses': False},'IE')

nest.Connect(pop1_nodes_in,pop1_nodes_in,{'rule': 'fixed_indegree','indegree': CII, 'autapses': False},'II')

##### POPULATION 2
nest.Connect(pop2_nodes_ex,pop2_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE, 'autapses': False},'EE')

nest.Connect(pop2_nodes_ex,pop2_nodes_in,{'rule': 'fixed_indegree','indegree': CEI, 'autapses': False},'EI')

nest.Connect(pop2_nodes_in,pop2_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE, 'autapses': False},'IE')

nest.Connect(pop2_nodes_in,pop2_nodes_in,{'rule': 'fixed_indegree','indegree': CII, 'autapses': False},'II')

##### POPULATION 3
nest.Connect(pop3_nodes_ex,pop3_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE, 'autapses': False},'EE')

nest.Connect(pop3_nodes_ex,pop3_nodes_in,{'rule': 'fixed_indegree','indegree': CEI, 'autapses': False},'EI')

nest.Connect(pop3_nodes_in,pop3_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE, 'autapses': False},'IE')

nest.Connect(pop3_nodes_in,pop3_nodes_in,{'rule': 'fixed_indegree','indegree': CII, 'autapses': False},'II')

##### POPULATION 4
nest.Connect(pop4_nodes_ex,pop4_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE, 'autapses': False},'EE')

nest.Connect(pop4_nodes_ex,pop4_nodes_in,{'rule': 'fixed_indegree','indegree': CEI, 'autapses': False},'EI')

nest.Connect(pop4_nodes_in,pop4_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE, 'autapses': False},'IE')

nest.Connect(pop4_nodes_in,pop4_nodes_in,{'rule': 'fixed_indegree','indegree': CII, 'autapses': False},'II')

##### POPULATION 5
nest.Connect(pop5_nodes_ex,pop5_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE, 'autapses': False},'EE')

nest.Connect(pop5_nodes_ex,pop5_nodes_in,{'rule': 'fixed_indegree','indegree': CEI, 'autapses': False},'EI')

nest.Connect(pop5_nodes_in,pop5_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE, 'autapses': False},'IE')

nest.Connect(pop5_nodes_in,pop5_nodes_in,{'rule': 'fixed_indegree','indegree': CII, 'autapses': False},'II')

##### POPULATION 6
nest.Connect(pop6_nodes_ex,pop6_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE, 'autapses': False},'EE')

nest.Connect(pop6_nodes_ex,pop6_nodes_in,{'rule': 'fixed_indegree','indegree': CEI, 'autapses': False},'EI')

nest.Connect(pop6_nodes_in,pop6_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE, 'autapses': False},'IE')

nest.Connect(pop6_nodes_in,pop6_nodes_in,{'rule': 'fixed_indegree','indegree': CII, 'autapses': False},'II')

##### POPULATION 7
nest.Connect(pop7_nodes_ex,pop7_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE, 'autapses': False},'EE')

nest.Connect(pop7_nodes_ex,pop7_nodes_in,{'rule': 'fixed_indegree','indegree': CEI, 'autapses': False},'EI')

nest.Connect(pop7_nodes_in,pop7_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE, 'autapses': False},'IE')

nest.Connect(pop7_nodes_in,pop7_nodes_in,{'rule': 'fixed_indegree','indegree': CII, 'autapses': False},'II')

##### POPULATION 8
nest.Connect(pop8_nodes_ex,pop8_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE, 'autapses': False},'EE')

nest.Connect(pop8_nodes_ex,pop8_nodes_in,{'rule': 'fixed_indegree','indegree': CEI, 'autapses': False},'EI')

nest.Connect(pop8_nodes_in,pop8_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE, 'autapses': False},'IE')

nest.Connect(pop8_nodes_in,pop8_nodes_in,{'rule': 'fixed_indegree','indegree': CII, 'autapses': False},'II')

##### POPULATION 9
nest.Connect(pop9_nodes_ex,pop9_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE, 'autapses': False},'EE')

nest.Connect(pop9_nodes_ex,pop9_nodes_in,{'rule': 'fixed_indegree','indegree': CEI, 'autapses': False},'EI')

nest.Connect(pop9_nodes_in,pop9_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE, 'autapses': False},'IE')

nest.Connect(pop9_nodes_in,pop9_nodes_in,{'rule': 'fixed_indegree','indegree': CII, 'autapses': False},'II')

##### POPULATION 10
nest.Connect(pop10_nodes_ex,pop10_nodes_ex,{'rule': 'fixed_indegree','indegree': CEE, 'autapses': False},'EE')

nest.Connect(pop10_nodes_ex,pop10_nodes_in,{'rule': 'fixed_indegree','indegree': CEI, 'autapses': False},'EI')

nest.Connect(pop10_nodes_in,pop10_nodes_ex,{'rule': 'fixed_indegree','indegree': CIE, 'autapses': False},'IE')

nest.Connect(pop10_nodes_in,pop10_nodes_in,{'rule': 'fixed_indegree','indegree': CII, 'autapses': False},'II')

##### POP1 ------>>>> POP2
nest.CopyModel('static_synapse_hom_w','bet_excitatory',{'weight':bet_g,'delay':bet_delay})
nest.CopyModel('static_synapse_hom_w','reso_excitatory',{'weight':bet_g_reso,'delay':bet_delay_reso})

nest.Connect(pop1_nodes_ex[1:NP],pop2_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

##### POP2 ------>>>> POP3
nest.Connect(pop2_nodes_ex[1:NP],pop3_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

##### POP3 ------>>>> POP4
nest.Connect(pop3_nodes_ex[1:NP],pop4_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

##### POP4 ------>>>> POP5
nest.Connect(pop4_nodes_ex[1:NP],pop5_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

##### POP5 ------>>>> POP6
nest.Connect(pop5_nodes_ex[1:NP],pop6_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

##### POP6 ------>>>> POP7
nest.Connect(pop6_nodes_ex[1:NP],pop7_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

##### POP7 ------>>>> POP8
nest.Connect(pop7_nodes_ex[1:NP],pop8_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

##### POP8 ------>>>> POP9
nest.Connect(pop8_nodes_ex[1:NP],pop9_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

##### POP9 ------>>>> POP10
nest.Connect(pop9_nodes_ex[1:NP],pop10_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

##### POP2 ------>>>> POP1  (Resonance)
nest.Connect(pop2_nodes_ex[1:NP],pop1_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'reso_excitatory')


nest.CopyModel('static_synapse_hom_w','pp_excitatory',{'weight':weight,'delay': 0.1})

nest.Connect(pp, pop1_nodes_ex[1:NP], {'rule':'fixed_indegree','indegree': CEE},'pp_excitatory')

nest.CopyModel('static_synapse_hom_w','NOISE_EXCI_SYN',{'weight':0.25,'delay':0.1})

nest.Connect(noise[:1], pop1_nodes_ex+pop2_nodes_ex+pop3_nodes_ex+ pop4_nodes_ex+pop5_nodes_ex+pop6_nodes_ex+pop7_nodes_ex+ pop8_nodes_ex+pop9_nodes_ex +pop10_nodes_ex, syn_spec='NOISE_EXCI_SYN')

nest.CopyModel('static_synapse_hom_w','NOISE_INHI_SYN',{'weight':0.4,'delay':0.1})

nest.Connect(noise[1:], pop1_nodes_in+pop2_nodes_in+pop3_nodes_in +pop4_nodes_in+pop5_nodes_in + pop6_nodes_in+ pop7_nodes_in  + pop8_nodes_in + pop9_nodes_in+ pop10_nodes_in, syn_spec='NOISE_INHI_SYN')

conns=nest.GetConnections(target=pop1_nodes_in)
conn_vals=nest.GetStatus(conns,"source")

spikes  = nest.Create('spike_detector',2,[{'label': 'resonance-py-ex'},{'label': 'resonance-py-in'}]) 

spikes_E= spikes[:1]
spikes_I= spikes[1:]

nest.Connect(nodes_ex,spikes_E)
nest.Connect(nodes_in,spikes_I)

vm = nest.Create('voltmeter', 1)

nest.Connect(vm, nodes_ex)

nest.Simulate(simtime)
'''
dmm   = nest.GetStatus(multimeter)[0]
Vms   = dmm["events"]["V_m"]
ts    = dmm["events"]["times"]
gexs  = dmm["events"]["g_ex"]
'''
spike_events = nest.GetStatus(spikes , 'n_events')
rate_ex= (spike_events[0]/simtime)*(1000.0/(10*NE))
rate_in= (spike_events[1]/simtime)*(1000.0/(10*NI))

Vm = nest.GetStatus(vm, 'events')[0]['V_m']
times = nest.GetStatus(vm, 'events')[0]['times']
senders = nest.GetStatus(vm, 'events')[0]['senders']

dSD = nest.GetStatus(spikes_E,keys="events")[0]
simtimes = np.arange(1, simtime)

Vm_single_10 = [Vm[senders == ii] for ii in pop10_nodes_ex[71:200]]
Vm_average_10 = np.mean(Vm_single_10, axis=0)

Vm_single_10  = [Vm[senders == ii] for ii in pop10_nodes_ex[71:200]]
Vm_average_10 = np.mean(Vm_single_10, axis=0)
Vm_single_9   = [Vm[senders == ii] for ii in pop9_nodes_ex[71:200]]
Vm_average_9  = np.mean(Vm_single_9, axis=0)
Vm_single_2   = [Vm[senders == ii] for ii in pop2_nodes_ex[71:200]]
Vm_average_2  = np.mean(Vm_single_2, axis=0)
Vm_single_1   = [Vm[senders == ii] for ii in pop1_nodes_ex[71:200]]
Vm_average_1  = np.mean(Vm_single_1, axis=0)

print("Excitatory rate: %.2f 1/sec" % rate_ex)
print("Inhibitory rate: %.2f 1/sec" % rate_in)

plt.subplot(4,1,1)
plt.plot(simtimes, Vm_average_10, 'b')

plt.xlabel('time (ms)', fontsize=18)

plt.subplot(4,1,2)
plt.plot(simtimes, Vm_average_9, 'b')
plt.subplot(4,1,3)
plt.plot(simtimes, Vm_average_2, 'b')
plt.ylabel('ave mem pot (mV)', fontsize=18)

plt.subplot(4,1,4)
plt.plot(simtimes, Vm_average_1, 'b')
plt.show()

nest.raster_plot.from_device(spikes_E , hist=True)
plt.xlabel('Time (ms)', fontsize=18)
plt.ylabel('Neuron ID', fontsize=18)
pylab.show()



