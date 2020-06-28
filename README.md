## Facilitating the propagation of spiking activity in feedforward networks by including feedback

#### Abstract:
Transient oscillations in network activity upon sensory stimulation have been reported in different sensory areas of the brain. These evoked oscillations are the generic response of networks of excitatory and inhibitory neurons (EI-networks) to a transient external input. Recently, it has been shown that this resonance property of EI-networks can be exploited for communication in modular neuronal networks by enabling the transmission of sequences of synchronous spike volleys ('pulse packets'), despite the sparse and weak connectivity between the modules. The condition for successful transmission is that the pulse packet (PP) intervals match the period of the modules' resonance frequency. Hence, the mechanism was termed communication through resonance (CTR). This mechanism has three severe constraints, though. First, it needs periodic trains of PPs, whereas single PPs fail to propagate. Second, the inter-PP interval needs to match the network resonance. Third, transmission is very slow, because in each module, the network resonance needs to build up over multiple oscillation cycles. Here, we show that, by adding appropriate feedback connections to the network, the CTR mechanism can be improved and the aforementioned constraints relaxed. Specifically, we show that adding feedback connections between two upstream modules, called the resonance pair, in an otherwise feedforward modular network can support successful propagation of a single PP throughout the entire network. The key condition for successful transmission is that the sum of the forward and backward delays in the resonance pair matches the resonance frequency of the network modules. The transmission is much faster, by more than a factor of two, than in the original CTR mechanism. Moreover, it distinctly lowers the threshold for successful communication by synchronous spiking in modular networks of weakly coupled networks. Thus, our results suggest a new functional role of bidirectional connectivity for the communication in cortical area networks.


### Usage
The python script 'SNR_Heat_Map_Plot.py' will reproduce the key results shown in the Figure 4 of the manuscript. Note that this may take some time. 
More code will be provided to reproduce other results. 

#### Requirements
- Nest Simulator (v2.18.0)
- pylab
- numpy
- matplotlib


### Citation
Rezaei H, Aertsen A, Kumar A, Valizadeh A (2020) Facilitating the propagation of spiking activity in feedforward networks by including feedback. To appear in PloS Comp. Biology. 

