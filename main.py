import libpymcr as lb
#mport numpy as np
m = lb.Matlab('C:/Users/jyw1/Desktop/SpinW/spinw/PySpinW.ctf')

J = 138.3
Jp = 2.0
Jpp = 2.0
Jc = 38.0

l = [J-Jc/2, Jp-Jc/4, Jpp]
divisor = 2
result = [divmod(x, divisor)[0] for x in l]

lacuo = m.sw_model('squareAF',result,0)
m.lacuo.unit_cell.S = 1/2
m.plot(lacuo,'range',[2, 2, 1])

Zc = 1.18

Qlist = ([3/4, 1/4, 0], [1/2, 1/2, 0], [1/2, 0, 0], [3/4, 1/4, 0], [1, 0, 0], [1/2, 0, 0], 100)
Qlab = ['P', 'M', 'X', 'P', '\Gamma', 'X']

lacuoSpec = lacuo.spinwave(Qlist,'hermit',False)
lacuoSpec["omega"] = lacuoSpec["omega"]*Zc

lacuoSpec = m.sw_neutron(lacuoSpec)
lacuoSpec = m.sw_egrid(lacuoSpec,'component','Sperp')
m.figure()
m.subplot(2,1,1)
m.sw_plotspec(lacuoSpec,'mode',3,'axLim',[0, 5],'dE',35,'dashed',True,'qlabel',Qlab)
m.colorbar('off', nargout=0)
m.subplot(2,1,2)
lacuoSpec = m.sw_omegasum(lacuoSpec,'zeroint',1e-5,'tol',1e-3)
m.sw_plotspec(lacuoSpec,'mode',2,'axLim',[0, 20],'dashed',True,'colormap',[0, 0, 0],'qlabel',Qlab)
m.swplot.subfigure(1,3,1)