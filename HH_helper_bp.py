import numpy as np
import brainpy as bp
import brainpy.math as bm
from scipy import stats as spstats

class HH_sbi(bp.NeuGroup):
    def __init__(self, size, ENa=53., gNa=50., EK=-107., gK=5., EL=-70., gL=0.1,
                 V_th= 10., C=1.0, gM=0.07, tau_max=6e2, Vt = -60.0, noise_factor = 0.1, **kwargs):
        # providing the group "size" information
        super(HH_sbi, self).__init__(size=size, **kwargs)
        
        # initialize parameters from HHsimulator
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.C = C
        self.V_th = V_th
        self.gM = gM
        self.tau_max = tau_max
        self.Vt = Vt
        # self.nois_fact = 0.1  # noise factor
        self.noise     = noise_factor

        # initialize variables
        self.V = bm.Variable(bm.random.randn(self.num) - 70.)
        self.m = bm.Variable(0.00168 * bm.ones(self.num))
        self.h = bm.Variable(0.99968 * bm.ones(self.num))
        self.n = bm.Variable(0.00654 * bm.ones(self.num))
        self.p = bm.Variable(0.02931 * bm.ones(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        # integral functions
        self.int_V = bp.odeint(f=self.dV, method='exp_auto')
        # self.int_V = bp.sdeint(f=self.dV, g =self.dg, method='exp_auto')
        self.int_m = bp.odeint(f=self.dm, method='exp_auto')
        self.int_h = bp.odeint(f=self.dh, method='exp_auto')
        self.int_n = bp.odeint(f=self.dn, method='exp_auto')
        self.int_p = bp.odeint(f=self.dp, method='exp_auto')

    def dV(self, V, t, m, h, n, p, Iext):
        I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
        I_K = (self.gK * n ** 4.0) * (V - self.EK)
        I_leak = self.gL * (V - self.EL)
        I_M = self.gM * p * (V - self.EK)
        dVdt = (- I_Na - I_K - I_leak - I_M + Iext) / self.C
        return dVdt

    def dg(self, V, t):
        return self.noise

    def dm(self, m, t, V):
        v1    = -0.25 * (V - self.Vt - 13.)
        alpha = 0.32 * (v1/(bm.exp(v1)-1)) / 0.25
        v2    = 0.2 * (V - self.Vt - 40.)
        beta  = 0.28 * (v2/(bm.exp(v2)-1)) / 0.2
        dmdt = alpha * (1 - m) - beta * m
        return dmdt

    def dh(self, h, t, V):
        v1    = V - self.Vt - 17.0
        alpha = 0.128 * bm.exp(-v1 / 18.0)
        v2    = V - self.Vt - 40.
        beta = 4.0 / (1 + bm.exp(-0.2 * v2))
        dhdt = alpha * (1 - h) - beta * h
        return dhdt

    def dn(self, n, t, V):
        v1    = -0.2 * (V - self.Vt - 15.0)
        alpha = 0.032 * (v1/(bm.exp(v1) - 1)) / 0.2
        v2    = V - self.Vt - 10.0
        beta =  0.5 * bm.exp(-v2 / 40)
        dndt = alpha * (1 - n) - beta * n
        return dndt
    
    def p_inf(self,V):
        v1 = V + 35.0
        return 1.0 / (1.0 + bm.exp(-0.1 * v1))
    
    def tau_p(self,V):
        v1 = V + 35.0
        return self.tau_max / (3.3 * bm.exp(0.05 * v1) + bm.exp(-0.05 * v1))
    
    def dp(self, p, t, V):
        p_inf = self.p_inf(V)
        tau_p = self.tau_p(V)
        dpdt = (p_inf - p) / tau_p
        return dpdt

    def update(self, tdi, x=None):
        _t, _dt = tdi.t, tdi.dt
        # compute V, m, h, n, p at the next time step
        noise_add    = self.noise * bm.random.randn(self.num) / bm.sqrt(_dt)
        V            = self.int_V(self.V, _t, self.m, self.h, self.n, self.p, self.input + noise_add, dt=_dt)
        self.h.value = self.int_h(self.h, _t, self.V, dt=_dt)
        self.m.value = self.int_m(self.m, _t, self.V, dt=_dt)
        # self.m.value = self.int_m(self.m, _t, self.V, dt=_dt)
        self.n.value = self.int_n(self.n, _t, self.V, dt=_dt)
        self.p.value = self.int_p(self.p, _t, self.V, dt=_dt)

        # update the spiking state and the last spiking time
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)

        # update V
        self.V.value = V

        # reset the external input
        self.input[:] = 0.


class Body_Wall_muscle(bp.NeuGroup):
    def __init__(self, size, ECa= 60., gCa= 15.6, EK=-40., gK=34., EL=-24, gL=0.1,
                 V_th= 10., C= 22, p_max = 0.1, phi= 0.04, phi_m = 1.2, gkr = 3.2, g_slo2 = 2.1 , g_Na = 0.1, ENa = 15., phi_n = 1.2, noise_factor = 0.01, **kwargs):
        # providing the group "size" information
        super(Body_Wall_muscle, self).__init__(size=size, **kwargs)

        # initialize parameters
        self.ECa = ECa
        self.EK = EK
        self.EL = EL
        self.ENa = ENa
        self.gCa = gCa
        self.g_Na   = g_Na
        self.gK = gK
        self.gL = gL
        self.C = C
        self.p_max = p_max
        self.V_th  = V_th
        self.noise =  noise_factor 
        self.phi_m  = phi_m
        self.phi_n  = phi_n
        self.alpha  = 43.
        self.beta   = 0.09
        self.g_slo2 = g_slo2
        self.gkr    = gkr
        self.phi    = phi

        # initialize variables
        self.V = bm.Variable(bm.random.randn(self.num) - 30.)
        self.m = bm.Variable(0.01 * bm.ones(self.num))
        self.h = bm.Variable(0.6 * bm.ones(self.num))
        self.n = bm.Variable(0.99 * bm.ones(self.num))
        self.p = bm.Variable(0.2 * bm.ones(self.num))
        self.kr = bm.Variable(0.0 * bm.ones(self.num))

        self.p_slo2 = bm.Variable(bm.zeros(self.num))
        self.Ca   = bm.Variable(bm.zeros(self.num))

        self.input = bm.Variable(bm.zeros(self.num))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        # integral functions
        self.int_V = bp.odeint(f=self.dV, method='exp_auto')
        self.int_m = bp.odeint(f=self.dm, method='exp_auto')
        self.int_h = bp.odeint(f=self.dh, method='exp_auto')
        self.int_n = bp.odeint(f=self.dn, method='exp_auto')
        self.int_p = bp.odeint(f=self.dp, method='exp_auto')
        self.int_p_slo2 = bp.odeint(f=self.dp_slo2, method='exp_auto')
        self.int_Ca = bp.odeint(f=self.dCa, method='exp_auto')
        self.int_kr = bp.odeint(f=self.dkr, method='exp_auto')

    def dV(self, V, t, m, h, n, p, p_slo2, kr, Iext):
        I_Ca = (self.gCa * m ** 2.0 * h) * (V - self.V_th - self.ECa)
        I_K = (self.gK * n ** 4.0) * (V - self.V_th - self.EK)
        I_M = (self.p_max * p) * (V - self.V_th - self.EK)
        I_slo2 = (self.g_slo2 * p_slo2) * (V -  self.EK)
        I_Na = self.g_Na * (V - self.V_th - self.ENa)
        I_kr = self.gkr *(1-kr) * self.krinf(V) *  (V - self.EK)
        I_leak = self.gL * (V - self.V_th  - self.EL)
        dVdt = (- I_Ca  - I_K - I_Na - I_slo2 - I_leak - I_kr - I_M + Iext) / self.C
        return dVdt
    
    krinf  = lambda self, V: 0.5 *(1+bm.tanh((V -  self.V_th + 42)/ 5.0))

    def dkr(self, kr, t, V):
        # krinf = 0.5 *(1+bm.tanh((V -  self.V_th + 42)/ 5.0))
        taumkr= 62
        dkrdt = (self.krinf(V)-kr)/taumkr
        return dkrdt

    def dp_slo2(self, p_slo2, t, Ca, V):
        C2 = self.alpha * bm.power(Ca, 2)
        C3 = C2 + self.beta
        return self.phi * (C2 / C3 - p_slo2) * C3

    def dCa(self, Ca, t, m, h, V):
        ICa = (self.gCa * m ** 2.0 * h) * (V - self.V_th - self.ECa)
        return -0.15 * ICa * 1e-4 - 0.075 * (Ca - 0.001)

    def dn(self, n, t, V):
        ninf = 0.5 * (bm.tanh((V - self.V_th +15.2)/36.22)+1)
        tau_n = 1.18+511.78/(1+bm.exp((V - self.V_th + 89.3)/21.92))
        dndt = self.phi_n * (ninf-n)/tau_n
        return dndt

    # def dm(self, m, t, V):
    #     tau_m = 61/(1+bm.exp((V - self.V_th + 81.2)/45.6)) + 22.39/(1+bm.exp(-(V - self.V_th -24.26)/22.26)) - 14.25 
    #     minf = -0.53/(1+bm.exp(-(V - self.V_th - 26)/6.4)) + 1.058/(1+bm.exp(-(V - self.V_th +8.75)/7.2655)) + 0.0095
    #     dmdt = self.phi_m * (minf-m)/tau_m
    #     return dmdt

    def dm(self, m, t, V):
        tau_m = 0.4 + .7 / (bm.exp(-(V + 5. - self.V_th) / 15.) +
                       bm.exp((V + 5. - self.V_th) / 15.))
        minf = 1. / (1 + bm.exp(-(V + 8. - self.V_th) / 8.6))
        dmdt = self.phi_m * (minf-m)/tau_m
        return dmdt

    def dh(self, h, t, V):
        # hinf = 0.435/(1+bm.exp((V  - self.V_th + 10.38)/0.5554)) + 64.045/(1+bm.exp(-(V  - self.V_th -171.5)/30.8)) + 0.1
        hinf   = 0.42 / (1. + bm.exp((V + 11. - self.V_th) / 2.)) + 0.28
        # hinf  = (1.43 / (1 + bm.exp(-(V - self.V_th + 15 - 14.9) / 12)) + 0.14) * (5.96 / (1 + bm.exp((V  - self.V_th  + 15 + 20.5) / 8.1)) + 0.6 - 0.32)
        tau_h = 24
        dhdt = (hinf-h)/tau_h
        return dhdt

    def dp(self, p, t, V):
        pinf = 1/(1+bm.exp(-(V- self.V_th +45)/10))
        tau_p = 4000/(3.38*bm.exp((V- self.V_th+45)/20)+bm.exp(-(V- self.V_th +45)/20))
        dpdt = (pinf-p)/tau_p
        return dpdt

    def update(self, tdi, x=None):
        _t, _dt = tdi.t, tdi.dt
        # compute V, m, h, n
        noise_add = self.noise * bm.random.randn(self.num) / bm.sqrt(_dt)
        V = self.int_V(self.V, _t, self.m, self.h, self.n, self.p, self.p_slo2, self.kr, self.input/0.75, dt=_dt)
        self.h.value = self.int_h(self.h, _t, self.V, dt=_dt)
        self.m.value = self.int_m(self.m, _t, self.V, dt=_dt)
        self.n.value = self.int_n(self.n, _t, self.V, dt=_dt)
        self.p.value = self.int_p(self.p, _t, self.V, dt=_dt)
        self.p_slo2.value = self.int_p_slo2(self.p_slo2, _t, self.Ca, self.V, dt=_dt)
        self.Ca.value = self.int_Ca(self.Ca, _t, self.m, self.h, self.V, dt=_dt)
        self.kr.value = self.int_kr(self.kr, _t, self.V, dt=_dt)

        # update the spiking state and the last spiking time
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)

        # update V
        self.V.value = V

        # reset the external input
        self.input[:] = 0.


# class Body_Wall_muscle(bp.NeuGroup):
#     def __init__(self, size, ECa= 60., gCa= 16.8, EK=-40., gK= 37., EL=-24, gL=0.1,
#                  V_th= 10., C= 22, p_max = 0.1, phi=1., phi_m = 1.2, g_slo2 = 10. , phi_n = 1.2, noise_factor = 0.01, **kwargs):
#         # providing the group "size" information
#         super(Body_Wall_muscle, self).__init__(size=size, **kwargs)

#         # initialize parameters
#         self.ECa = ECa
#         self.EK = EK
#         self.EL = EL
#         self.gCa = gCa
#         self.gK = gK
#         self.gL = gL
#         self.C = C
#         self.p_max = p_max
#         self.V_th  = V_th
#         self.noise =  noise_factor 
#         self.phi_m  = phi_m
#         self.phi_n  = phi_n
#         self.g_Na   = 0.01
#         self.ENa    = 54.4
#         self.alpha  = 43.
#         self.beta   = 0.09
#         self.g_slo2 = g_slo2
#         self.phi    = phi

#         # initialize variables
#         self.V = bm.Variable(bm.random.randn(self.num) - 30.)
#         self.m = bm.Variable(0.01 * bm.ones(self.num))
#         self.h = bm.Variable(0.6 * bm.ones(self.num))
#         self.n = bm.Variable(0.82 * bm.ones(self.num))
#         self.p = bm.Variable(0.2 * bm.ones(self.num))

#         self.p_slo2 = bm.Variable(bm.zeros(self.num))
#         self.Ca   = bm.Variable(bm.zeros(self.num))

#         self.input = bm.Variable(bm.zeros(self.num))
#         self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
#         self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

#         # integral functions
#         self.int_V = bp.odeint(f=self.dV, method='exp_auto')
#         self.int_m = bp.odeint(f=self.dm, method='exp_auto')
#         self.int_h = bp.odeint(f=self.dh, method='exp_auto')
#         self.int_n = bp.odeint(f=self.dn, method='exp_auto')
#         self.int_p = bp.odeint(f=self.dp, method='exp_auto')
#         self.int_p_slo2 = bp.odeint(f=self.dp_slo2, method='exp_auto')
#         self.int_Ca = bp.odeint(f=self.dCa, method='exp_auto')

#     def dV(self, V, t, m, h, n, p, p_slo2, Iext):
#         I_Ca = (self.gCa * m ** 2.0 * h) * (V - self.V_th - self.ECa)
#         I_K = (self.gK * n ** 4.0) * (V - self.V_th - self.EK)
#         I_M = (self.p_max * p) * (V - self.V_th - self.EK)
#         I_slo2 = (self.g_slo2 * p_slo2) * (V -  self.EK)
#         I_Na = self.g_Na * (V - self.V_th - self.ENa)
#         I_leak = self.gL * (V - self.V_th  - self.EL)
#         dVdt = (- I_Ca  - I_K - I_Na - I_slo2 - I_leak - I_M + Iext) / self.C
#         return dVdt

#     def dp_slo2(self, p_slo2, t, Ca, V):
#         C2 = self.alpha * bm.power(Ca, 2)
#         C3 = C2 + self.beta
#         return self.phi * (C2 / C3 - p_slo2) * C3

#     def dCa(self, Ca, t, m, h, V):
#         ICa = (self.gCa * m ** 2.0 * h) * (V - self.V_th - self.ECa)
#         return -0.15 * ICa * 1e-4 - 0.075 * Ca

#     def dn(self, n, t, V):
#         ninf = 0.5 * (bm.tanh((V - self.V_th +15.2)/36.22)+1)
#         tau_n = 1.18+511.78/(1+bm.exp((V - self.V_th + 89.3)/21.92))
#         dndt = self.phi_n * (ninf-n)/tau_n
#         return dndt

#     def dm(self, m, t, V):
#         tau_m = 61/(1+bm.exp((V - self.V_th + 81.2)/45.6)) + 22.39/(1+bm.exp(-(V - self.V_th -24.26)/22.26)) - 14.25 
#         minf = -0.53/(1+bm.exp(-(V - self.V_th - 26)/6.4)) + 1.058/(1+bm.exp(-(V - self.V_th +8.75)/7.2655)) + 0.0095
#         dmdt = self.phi_m * (minf-m)/tau_m
#         return dmdt

#     def dh(self, h, t, V):
#         hinf = 0.435/(1+bm.exp((V  - self.V_th + 10.38)/0.5554)) + 64.045/(1+bm.exp(-(V  - self.V_th -171.5)/30.8)) + 0.1
#         # hinf  = (1.43 / (1 + bm.exp(-(V - self.V_th + 15 - 14.9) / 12)) + 0.14) * (5.96 / (1 + bm.exp((V  - self.V_th  + 15 + 20.5) / 8.1)) + 0.6 - 0.32)
#         tau_h = 20
#         dhdt = (hinf-h)/tau_h
#         return dhdt

#     def dp(self, p, t, V):
#         pinf = 1/(1+bm.exp(-(V- self.V_th +45)/10))
#         tau_p = 4000/(3.38*bm.exp((V- self.V_th+45)/20)+bm.exp(-(V- self.V_th +45)/20))
#         dpdt = (pinf-p)/tau_p
#         return dpdt

#     def update(self, tdi, x=None):
#         _t, _dt = tdi.t, tdi.dt
#         # compute V, m, h, n
#         noise_add = self.noise * bm.random.randn(self.num) / bm.sqrt(_dt)
#         V = self.int_V(self.V, _t, self.m, self.h, self.n, self.p, self.p_slo2, self.input/0.75 + noise_add, dt=_dt)
#         self.h.value = self.int_h(self.h, _t, self.V, dt=_dt)
#         self.m.value = self.int_m(self.m, _t, self.V, dt=_dt)
#         self.n.value = self.int_n(self.n, _t, self.V, dt=_dt)
#         self.p.value = self.int_p(self.p, _t, self.V, dt=_dt)
#         self.p_slo2.value = self.int_p_slo2(self.p_slo2, _t, self.Ca, self.V, dt=_dt)
#         self.Ca.value = self.int_Ca(self.Ca, _t, self.m, self.h, self.V, dt=_dt)

#         # update the spiking state and the last spiking time
#         self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
#         self.t_last_spike.value = bm.where(self.spike, _t, self.t_last_spike)

#         # update V
#         self.V.value = V

#         # reset the external input
#         self.input[:] = 0.