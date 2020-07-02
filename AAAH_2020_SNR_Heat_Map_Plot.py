# Copyright 2020 Hedyeh Rezaei
# The MIT License

import nest
import pylab
import nest.raster_plot

import time
import numpy as np
import matplotlib.pyplot as plt


msd=1000
n_vp=nest.GetKernelStatus('total_num_virtual_procs')
msdrange1=range(msd,msd+n_vp)

pyrngs=[numpy.random.RandomState(s) for s in msdrange1]
msdrange2=range(msd+n_vp+1,msd+1+2*n_vp)
nest.SetKernelStatus({'grng_seed':msd+n_vp , 'rng_seeds':msdrange2})

bin_w = 5.
time_range1 = (350.,750.)

a = 25                 # number of spikes in one pulse packet
sdev = 2.0             # width of pulse packet (ms)
weight = 0.33          # PP amplitude (mV)
pulse_times = [800.]   # occurrence time (center) of pulse-packet (ms)

NP        = 70
NE        = 200             # number of excitatory neurons
NI        = 50              # number of inhibitory neurons
N_neurons = NE+NI           # number of neurons in total
N_rec     = 10*N_neurons    # record from all neuron

simtime    = 1600.     # Simulation time in ms

in_delay   = 1.5       # within-layer synaptic delay in ms

p_rate_ex = 8000.0
p_rate_in = p_rate_ex - 1700.0

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
  
bet_g_arr = np.arange(0.29, 0.35, 0.01)
bet_delay_arr = np.arange(2.0, 20., 0.5)

SNR_arr = np.zeros((bet_g_arr.shape[0], bet_delay_arr.shape[0]))


for ii,bet_delay in enumerate(bet_delay_arr): 
    for jj,bet_g in enumerate(bet_g_arr): 

        nest.ResetKernel()
        nest.SetKernelStatus({"overwrite_files": True})
        time_range2 = (800. + 10*bet_delay,1200. + 10*bet_delay)

        nest.CopyModel("static_synapse","EI",{"weight":j_exc_inh,"delay":in_delay})
        nest.CopyModel("static_synapse","EE",{"weight":j_exc_exc,"delay":in_delay})
        nest.CopyModel("static_synapse","IE",{"weight":j_inh_exc,"delay":in_delay})
        nest.CopyModel("static_synapse","II",{"weight":j_inh_inh,"delay":in_delay})

        nest.CopyModel('static_synapse_hom_w','bet_excitatory',{'weight':bet_g,'delay':bet_delay})
        
        nest.CopyModel('static_synapse_hom_w','NOISE_INHI_SYN',{'weight':0.4,'delay':0.1})
        nest.CopyModel('static_synapse_hom_w','pp_excitatory',{'weight':weight,'delay': 0.1})
        nest.CopyModel('static_synapse_hom_w','NOISE_EXCI_SYN',{'weight':0.25,'delay':0.1})

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

        nodes= pop1_nodes_ex + pop1_nodes_in + pop2_nodes_ex + pop2_nodes_in + pop3_nodes_ex + pop3_nodes_in + pop4_nodes_ex + pop4_nodes_in + pop5_nodes_ex + pop5_nodes_in + pop6_nodes_ex + pop6_nodes_in + pop7_nodes_ex + pop7_nodes_in +   pop8_nodes_ex + pop8_nodes_in + pop9_nodes_ex + pop9_nodes_in + pop10_nodes_ex + pop10_nodes_in

        nodes_ex= pop1_nodes_ex + pop2_nodes_ex + pop3_nodes_ex + pop4_nodes_ex + pop5_nodes_ex + pop6_nodes_ex + pop7_nodes_ex + pop8_nodes_ex + pop9_nodes_ex + pop10_nodes_ex

        nodes_in= pop1_nodes_in + pop2_nodes_in + pop3_nodes_in + pop4_nodes_in + pop5_nodes_in + pop6_nodes_in + pop7_nodes_in + pop8_nodes_in + pop9_nodes_in + pop10_nodes_in 

        node_info=nest.GetStatus(nodes)
        local_nodes=[(ni['global_id'],ni['vp']) 
                     for ni in node_info if ni ['local']]
        for gid,vp in local_nodes:
            nest.SetStatus([gid],{'V_m':pyrngs[vp].uniform(-75.0,-65.0)})

        pp = nest.Create('pulsepacket_generator',params={'activity':a,'sdev':sdev,'pulse_times':pulse_times})

        noise = nest.Create("poisson_generator", 2, [{"rate": p_rate_ex}, {"rate": p_rate_in}])

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
        nest.Connect(pop1_nodes_ex[1:NP],pop2_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

        ##### POP2 ------>>>> POP3
        nest.Connect(pop2_nodes_ex[NP+1:2*NP],pop3_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

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
        nest.Connect(pop2_nodes_ex[1:NP],pop1_nodes_ex[1:NP],{'rule': 'fixed_indegree','indegree': CEP},'bet_excitatory')

        nest.Connect(pp, pop1_nodes_ex[1:NP], {'rule': 'fixed_indegree','indegree': CEE},'pp_excitatory')

        nest.Connect(noise[:1], pop1_nodes_ex+pop2_nodes_ex+pop3_nodes_ex+ pop4_nodes_ex+pop5_nodes_ex+pop6_nodes_ex+pop7_nodes_ex+ pop8_nodes_ex+pop9_nodes_ex +pop10_nodes_ex, syn_spec='NOISE_EXCI_SYN')

        nest.Connect(noise[1:], pop1_nodes_in+pop2_nodes_in+pop3_nodes_in +pop4_nodes_in+pop5_nodes_in+pop6_nodes_in+ pop7_nodes_in  + pop8_nodes_in + pop9_nodes_in+ pop10_nodes_in, syn_spec='NOISE_INHI_SYN')

        spikes  = nest.Create('spike_detector',2,[{'label': 'resonance-py-ex'},{'label': 'resonance-py-in'}]) 
        spikes_E= spikes[:1]
        spikes_I= spikes[1:]


        nest.SetStatus(spikes_E,[{"label": "B",
                         "withtime": True,
                         "withgid": True,
                         "to_file": True}])

        nest.Connect(pop10_nodes_ex,spikes_E)
        nest.Connect(nodes_in,spikes_I)


        nest.Simulate(simtime)
        
        act = np.loadtxt('B-2504-0.gdf')
        evs = act[:,0]
        ts = act[:,1]
        
        if time_range1!=[]:
            idx1 = (ts>time_range1[0]) & (ts<=time_range1[1])
            spikes1 = ts[idx1]

        if len(spikes1) == 0:
           print('psd: spike array is empty')

        if time_range2!=[]:
            idx2 = (ts>time_range2[0]) & (ts<=time_range2[1])
            spikes2 = ts[idx2]

        if len(spikes2) == 0:
           print('psd: spike array is empty')

        ids = np.unique(evs)
        nr_neurons = len(ids)

        bins1 = np.arange(time_range1[0],time_range1[1],bin_w)
        bins2 = np.arange(time_range2[0],time_range2[1],bin_w)

        a1,b1 = np.histogram(spikes1, bins1)
        a2,b2 = np.histogram(spikes2, bins2)

        VAR_O_10  = np.var(a1, axis=0)
        VAR_A_10  = np.var(a2, axis=0)
        #To obtain SNR of tenth layer

        SNR_arr[jj,ii]  = (VAR_A_10/VAR_O_10)


plt.figure()
X,Y = np.meshgrid(bet_delay_arr,bet_g_arr)
plt.pcolor(X,Y,SNR_arr)
plt.colorbar()
pylab.show()

