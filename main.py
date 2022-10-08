'''
Centroid Molecular Dynamics (CMD) - a vairant of PIMD
Author: Wenbin FAN @Fudan
Advisor: Wei FANG @Fudan
Date: Oct. 8, 2022

All atomic unit used in program. Only time in input line is fs.
'''

import numpy as np
import matplotlib.pyplot as plt
import constant
import element

class simple_cmd():

    def pot(self, q=None):
        '''
        Morse function for OH bond
        '''
        if q is None: q = self.q

        D = 0.16253
        k = 0.59192
        r0 = 1.80377 # in Bohr
        # r0 = 100
        alpha = np.sqrt(k / D / 2.0)

        rs = np.sqrt(np.sum(np.power(q[:, 1, :] - q[:, 0, :], 2), axis=0))  # r_start
        r = rs - r0
        r *= alpha
        e = np.exp(-r)  # exp(- alpha *(x-r0) )

        # potential
        v = D * (1 - e) ** 2 - D
        # gradient
        dvdr = -2 * D * alpha * e * (e - 1) # Nbead array

        dvdx = np.zeros([3, self.Natom, self.Nbead])
        dvdx[:, 0, :] = (q[:, 0, :] - q[:, 1, :])# * constant.bohr2ang
        dvdx[:, 1, :] = -dvdx[:, 0, :]
        for k in range(self.Nbead):
            dvdx[:,:,k] *= dvdr[k] / rs[k]

        return v, dvdx

    def __init__(self,
                 temperature=300.0,
                 Nbead=16,
                 Nstep=1000,
                 dt=0.1,
                 gamma=10.0):

        self.Natom = 2  # hard-coded here for OH radical only
        self.Nbead = Nbead
        self.Nstep = Nstep
        self.dt = dt / constant.au2fs  # to au
        self.gamma = gamma  # mass scale for non-centroid motion

        self.element = ['H', 'O']

        self.T_Kelvin = temperature
        self.beta = constant.au2J / (constant.kB * self.T_Kelvin)
        self.beta_n = self.beta / Nbead

        # get mass in au
        mass = [element.atomicMass[self.element[i]] for i in range(self.Natom)]
        self.mass = np.array(mass) * constant.amu2me

        self.q = np.zeros([3, self.Natom, self.Nbead])  # coordinate in Bohr
        self.p = np.zeros([3, self.Natom, self.Nbead])  # momentum in au

        self.init_Cmat(self.Nbead)

        self.sample_momentum()

        self.mass_scale = np.zeros(self.Nbead)
        self.mass_scale[:] = 1.0
        if self.gamma > 1:
            self.mass_scale[1:] = 1 / self.gamma

        self.omega_k = np.zeros(self.Nbead)
        for k in range(self.Nbead):
            self.omega_k[k] = 2.0 / self.beta_n * np.sin(k * np.pi / self.Nbead)

        return

    def __fft(self, x):
        # N = len(x)
        # Nhalf = int(np.ceil(N / 2))
        # xs = np.fft.fft(x, norm="ortho")# / np.sqrt(N)
        #
        # xs_real = np.zeros(N)
        # xs_real[:Nhalf + 1] = np.real(xs[:Nhalf + 1])
        # xs_real[Nhalf + 1:] = -np.imag(xs[Nhalf + 1:])
        #
        # return xs_real
        return np.dot(x, self.transformC)

    def __ifft(self, xs):
        # N = len(xs)
        # Nhalf = int(np.ceil(N / 2))
        # x = np.zeros(N, dtype=complex)
        #
        # for i in range(1, Nhalf):
        #     x[i] = complex(xs[i], xs[N - i])
        # for i in range(Nhalf + 1, N):
        #     x[i] = complex(xs[N - i], -xs[i])
        # x[0] = xs[0]
        # x[Nhalf] = xs[Nhalf]
        # x_ifft = np.fft.ifft(x, norm="ortho")
        # x_real = np.real(x_ifft)
        #
        # return x_real# * np.sqrt(N)
        return np.dot(self.transformC, xs)

    def init_Cmat(self, Nbead):

        c = np.zeros([Nbead, Nbead])

        for j in range(Nbead):
            for k in range(Nbead):
                if k == 0:
                    c[j, k] = 1
                elif k >= 1 and k <= int(Nbead / 2 - 1):
                    c[j, k] = np.sqrt(2) * np.cos(2 * np.pi * j * k / Nbead)
                elif k == int(Nbead / 2):
                    c[j, k] = (-1) ** j
                elif k >= int(Nbead / 2 + 1) and k <= Nbead - 1:
                    c[j, k] = np.sqrt(2) * np.sin(2 * np.pi * j * k / Nbead)
                else:
                    print(f'wrong range of i,j {j},{k}', )
                    exit()

        self.transformC = c / np.sqrt(self.Nbead)

        return

    def free_ring_polymer(self):

        ps = np.zeros([3, self.Natom, self.Nbead])
        qs = np.zeros([3, self.Natom, self.Nbead])
        for i in range(3):
            for j in range(self.Natom):
                qs[i, j, :] = self.__fft(self.q[i, j, :])
                ps[i, j, :] = self.__fft(self.p[i, j, :])

        for j in range(self.Natom):
            poly = np.zeros([4, self.Nbead])
            poly[0, 0] = 1.0
            poly[2, 0] = self.dt / self.mass[j]
            poly[3, 0] = 1.0

            for k in range(1, self.Nbead):
                omegak = self.omega_k[k] #2.0 / self.beta_n * np.sin(k * np.pi / self.Nbead)
                omegat = omegak * self.dt
                omegam = omegak * self.mass[j] * self.mass_scale[k]

                poly[0, k] = np.cos(omegat)
                poly[1, k] = - omegam * np.sin(omegat)
                poly[2, k] = np.sin(omegat) / omegam
                poly[3, k] = poly[0, k]

            for k in range(self.Nbead):
                for i in range(3):
                    p_new = ps[i,j,k] * poly[0,k] + qs[i,j,k] * poly[1,k]
                    q_new = ps[i,j,k] * poly[2,k] + qs[i,j,k] * poly[3,k]
                    ps[i,j,k] = p_new
                    qs[i,j,k] = q_new

        for i in range(3):
            for j in range(self.Natom):
                self.q[i, j, :] = self.__ifft(qs[i, j, :])
                self.p[i, j, :] = self.__ifft(ps[i, j, :])

        return

    def verlet(self):

        v, grad = self.pot(self.q)
        self.p -= self.dt * grad / 2.0

        if self.Nbead == 1:
            for j in range(self.Natom):
                self.q[:,j,:] += self.p[:,j,:] * self.dt / self.mass[j]
        else:
            self.free_ring_polymer()

        v, grad = self.pot(self.q)
        self.p -= self.dt * grad / 2.0

        return

    def sample_momentum(self):

        r = np.random.normal(loc=0, scale=1, size=[3, self.Natom, self.Nbead])  # random Gaussian momentum
        scale = np.sqrt(self.mass / self.beta_n)
        self.p = np.einsum('ijk,j -> ijk', r, scale)

        return

    def __get_kinetic_energy(self):
        Ek = 0.0
        if self.gamma <= 1:
            for j in range(self.Natom):
                Ek += np.sum(np.power(self.p[:,j,:], 2)) / self.mass[j] / 2.0
        else:
            for i in range(3):
                for j in range(self.Natom):
                    ps = self.__fft(self.p[i, j, :])
                    Ek += np.dot(np.power(ps, 2), 1/self.mass_scale) / self.mass[j] / 2.0
        return Ek / self.Nbead

    def __get_potential_energy(self):
        v, g = self.pot(self.q)
        return np.sum(v) / self.Nbead

    def __get_ring_energy(self):
        Ering = 0.0
        omega_n = self.Nbead / self.beta
        if self.gamma <= 1:
            for j in range(0, self.Natom):
                for k in range(0, self.Nbead):
                    b = (k+1)%self.Nbead
                    dx = self.q[0,j,k] - self.q[0,j,b]
                    dy = self.q[1,j,k] - self.q[1,j,b]
                    dz = self.q[2,j,k] - self.q[2,j,b]
                    Ering += self.mass[j] * omega_n * omega_n * (dx*dx + dy*dy + dz*dz) / 2.0
        else:
            for i in range(3):
                for j in range(0, self.Natom):
                    qs = self.__fft(self.q[i,j,:])
                    Ering += 0.5 * np.sum(self.omega_k**2 * np.power(qs,2) * self.mass_scale) * self.mass[j]
        return Ering / self.Nbead

    def __get_centroid(self):
        return np.mean(self.q, axis=2)

    def __get_radius_of_gyration(self):
        cen = self.__get_centroid()
        r = np.zeros(self.Natom)
        for j in range(self.Natom):
            for k in range(self.Nbead):
                dr = np.power(self.q[:,j,k] - cen[:,j], 2)
                r[j] += np.sqrt(np.sum(dr))
            r[j] /= self.Nbead
        return r

    def write_xyz(self, file, Natom, Nbead, ele, q):

        file.write(f'{Natom*Nbead}\n\n')
        for j in range(Natom):
            for k in range(Nbead):
                file.write('{} {:12.6f}  {:12.6f}  {:12.6f}\n'.format(ele[j], *q[:,j,k] * constant.bohr2ang))
        file.flush()

        return

    def main(self):
        f = open('traj.xyz', 'w')
        log_file = open('log.dat', 'w')

        self.q[0, 1, :] = 2 # / constant.bohr2ang
        e = []

        step = 0
        while step < self.Nstep:
            self.verlet()
            if step%100 == 0:
                self.write_xyz(file=f, Natom=self.Natom, Nbead=self.Nbead, q=self.q, ele=self.element)
            if step%10 == 0:
                Ek = self.__get_kinetic_energy()
                Ep = self.__get_potential_energy()
                Ering = self.__get_ring_energy()
                rg = self.__get_radius_of_gyration()
                Etot = Ek + Ep + Ering
                line = '{}\t{:14.8f}\t{:14.8f}\t{:14.8f}\t{:14.8f}\t{}'.format(step, Ek, Ep, Ering, Etot, rg)
                print(line)
                print(line, file=log_file)
                e.append([Ek, Ep, Ering, Etot])
            # if step%1000 == 0:
            #     print('Sample momentum! ')
            #     self.sample_momentum()

            step += 1

        e = np.array(e)

        plt.plot(e[:,0], color='black', linewidth=2, label='Ek')
        plt.plot(e[:,1], label='Ep')
        plt.plot(e[:,2], label='Ering')
        plt.plot(e[:,3], color='red', linewidth=2, label='Etot')
        plt.legend()
        plt.show()

        print(np.mean(e[:,3]), np.std(e[:,3]))
        log_file.close()
        f.close()

        return



if __name__ == '__main__':
    '''
    Temperature in Kelvin, time step in fs
    '''
    a = simple_cmd(temperature=300, Nbead=8, Nstep=10000, dt=0.01, gamma=10)
    a.main()

