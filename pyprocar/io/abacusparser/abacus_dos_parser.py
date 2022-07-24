# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 20:49:03 2022

@author: Yuyang Ji
"""

import os
import re
from turtle import position
import xml.etree.ElementTree as ET
from collections import OrderedDict
import numpy as np
from scipy.constants import physical_constants

from pyprocar.core import DensityOfStates, Structure

BOHR_TO_A = physical_constants["atomic unit of length"][0] / \
    physical_constants["Angstrom star"][0]


class ABACUSDOSParser:
    def __init__(self, pdos_file='PDOS', running_file='running_scf.log', dos_interpolation_factor=None):

        self.dos_interpolation_factor = dos_interpolation_factor

        if not os.path.isfile(pdos_file):
            raise ValueError('File not found ' + pdos_file)
        else:
            self.pdos_file = pdos_file

        if not os.path.isfile(running_file):
            raise ValueError('File not found ' + running_file)
        else:
            self.running_file = running_file

        self.nspin, self.eunit, self.state, self.energies = self.read()
        self._nsplit = self.nspin if self.nspin == 2 else 1

        if self.nspin in [1, 4]:
            self.is_spin_polarized = False
        elif self.nspin == 2:
            self.is_spin_polarized = True

        self._readABACUSout()

        self.orbital_name = [
            's',
            'p_x', 'p_y', 'p_z',
            'd_xy', 'd_zy', 'd_z^2', 'd_zx', 'd_x^2-y^2',
            'f_z^3', 'f_xz^2', 'f_yz^2', 'f_xyz', 'f_z(x^2-y^2)',
            'f_x(x^2-3y^2)', 'f_y(3x^2-y^2)',
            'g_z^4', 'g_xz^3', 'g_yz^3', 'g_xyz^2', 'g_z^2(x^2-y^2)',
            'g_x^3z', 'g_y^3z', 'g_x^4+y^4', 'g_xy(x^2-y^2)',
        ]

    def read(self):
        """Read partial DOS data file"""

        pdosdata = ET.parse(self.pdos_file)
        root = pdosdata.getroot()
        nspin = int(root.findall('.//nspin')[0].text.replace(' ', ''))
        norbitals = int(root.findall('.//norbitals')[0].text.replace(' ', ''))
        eunit = root.findall(
            './/energy_values')[0].attrib['units'].replace(' ', '')
        e_list = root.findall(
            './/energy_values')[0].text.replace(' ', '').split('\n')
        remove_empty(e_list)
        state = []
        for i in range(norbitals):
            orb = OrderedDict()
            orb['atom_index'] = int(root.findall(
                './/orbital')[i].attrib['atom_index'].replace(' ', ''))
            orb['species'] = root.findall(
                './/orbital')[i].attrib['species'].replace(' ', '')
            orb['l'] = int(root.findall('.//orbital')
                           [i].attrib['l'].replace(' ', ''))
            orb['m'] = int(root.findall('.//orbital')
                           [i].attrib['m'].replace(' ', ''))
            orb['z'] = int(root.findall('.//orbital')
                           [i].attrib['z'].replace(' ', ''))
            data = root.findall('.//data')[i].text.split('\n')
            data = handle_data(data)
            remove_empty(data)
            orb['data'] = np.array(data, dtype=float)
            state.append(orb)
        energies = np.array(e_list).astype(float)

        return nspin, eunit, state, energies

    def _get_dos_total(self):

        dos_total = {'energies': self.energies}
        res = np.zeros_like(self.state[0]["data"], dtype=float)
        for orb in self.state:
            res = res + orb['data']

        if self.is_spin_polarized:
            dos_total['Spin-up'] = res[:, 0]
            dos_total['Spin-down'] = res[:, 1]
        else:
            dos_total['Spin-up'] = res[:, 0]

        return dos_total, list(dos_total.keys())

    def _get_dos_projected(self):
        """dos_projected[name][l_index][m_index].shape == (obj._nsplit, len(obj.energies))"""

        dos_projected = dict()
        res = np.zeros((len(self.energies), self._nsplit))
        for orb in self.state:
            elem = orb['species']
            name = elem+str(orb['atom_index']-1)
            l_index = orb['l']
            m_index = orb['m']
            z_index = orb['z']
            zs = self.zetas[elem][l_index]
            if z_index <= zs:
                res += orb['data']
            if z_index == zs:
                dos_projected[name] = {'energies': self.energies}
                dos_projected[name][l_index] = {m_index: res.T}
                res = np.zeros((len(self.energies), self._nsplit))

        return dos_projected, self.orbital_name[:self._max_M]

    @property
    def dos(self):
        energies = self.dos_total['energies'] - self.fermi
        total = []
        for ispin in self.dos_total:
            if ispin == 'energies':
                continue
            total.append(self.dos_total[ispin])
        return DensityOfStates(
            energies=energies,
            total=total,
            projected=self.dos_projected,
            interpolation_factor=self.dos_interpolation_factor)

    @property
    def dos_to_dict(self):
        """
        Returns the complete density (total,projected) of states as a python dictionary
        """
        return {
            'total': self._get_dos_total(),
            'projected': self._get_dos_projected()
        }

    @property
    def dos_total(self):
        """
        Returns the total density of states as a pychemia.visual.DensityOfSates object
        """

        dos_total, labels = self._get_dos_total()

        return dos_total

    @property
    def dos_projected(self):
        """
        Returns the projected DOS as a multi-dimentional array, to be used in the
        pyprocar.core.dos object
        """
        pass

#     ###########################################################################
#     # This section parses for the projected density of states and puts it in a
#     # Pychemia Density of States Object
#     ###########################################################################

    @property
    def species(self):
        """
        Returns the species
        """
        return self.initial_structure.species

    @property
    def structures(self):
        """
        Returns a list of pychemia.core.Structure representing all the ionic step structures
        """

        structures = []
        for pos, cell in zip(self.positions, self.cells):
            if self.coord_class == 'CARTESIAN':
                st = Structure(atoms=self.labels,
                               cartesian_coordinates=pos, lattice=cell)
            elif self.coord_class == 'DIRECT':
                st = Structure(atoms=self.labels,
                               fractional_coordinates=pos, lattice=cell)
        structures.append(st)

        return structures

    @property
    def structure(self):
        """
        crystal structure of the last step
        """
        return self.structures[-1]

    @property
    def initial_structure(self):
        """
        Returns the initial Structure as a pychemia structure
        """
        return self.structures[0]

    @property
    def final_structure(self):
        """
        Returns the final Structure as a pychemia structure
        """

        return self.structures[-1]

    @property
    def fermi(self):
        """
        Returns the fermi energy read from running_*.log
        """

        return self._fermi

#     ###########################################################################
#     ###########################################################################
#     ###########################################################################

    def _readABACUSout(self):

        # Structures
        def str_to_sites(val_in):
            data = dict()
            val = np.array(val_in)
            labels = val[:, 0]
            pos = val[:, 1:4].astype(float)
            return labels, pos

        def str_to_cell(val_in):
            val = list(map(float, val_in))
            #val = [v.strip().split() for v in val_in.split('\n')]
            return np.reshape(val, (3, 3)).astype(float)*alat

        def format_steps(pattern):
            steps = list(map(int, pattern.findall(contents)))
            steps.insert(0, 0)
            return steps

        re_float = r'[\d\.\-\+Ee]+'
        fd = open(self.running_file)
        contents = fd.read()

        # cells
        a0_pattern = re.compile(
            rf'lattice constant \(Bohr\)\s*=\s*({re_float})')
        cell_pattern = re.compile(
            rf'Lattice vectors: \(Cartesian coordinate: in unit of a_0\)\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n\s*({re_float})\s*({re_float})\s*({re_float})\n')
        ionstep_pattern = re.compile(
            r'(?:STEP OF ION RELAXATION\s*:\s*|RELAX IONS\s*:\s*\d+\s*\(in total:\s*)(\d+)')
        cellstep_pattern = re.compile(r'RELAX CELL\s*:\s*(\d+)')
        alat = float(a0_pattern.search(contents).group(1))*BOHR_TO_A
        if 'RELAX CELL' in contents:  # cell-relax
            cell_steps = format_steps(cellstep_pattern)
            self.cells = np.array(list(map(str_to_cell, cell_pattern.findall(contents))))[
                [cell_steps]]
        elif 'STEP OF ION RELAXATION' in contents:   # relax
            ion_steps = format_steps(ionstep_pattern)
            self.cells = np.array(
                list(map(str_to_cell, cell_pattern.findall(contents)))*len(ion_steps))
        elif 'SELF-CONSISTENT' in contents:    # scf/nscf
            self.cells = np.array(
                list(map(str_to_cell, cell_pattern.findall(contents))))

        # labels and positions
        pos_pattern = re.compile(
            rf'(CARTESIAN COORDINATES \( UNIT = {re_float} Bohr \)\.+\n\s*atom\s*x\s*y\s*z\s*mag(\s*vx\s*vy\s*vz\s*|\s*)\n[\s\S]+?)\n\n|(DIRECT COORDINATES\n\s*atom\s*x\s*y\s*z\s*mag(\s*vx\s*vy\s*vz\s*|\s*)\n[\s\S]+?)\n\n')
        site_pattern = re.compile(
            rf'tau[cd]_([a-zA-Z]+)\d+\s+({re_float})\s+({re_float})\s+({re_float})\s+({re_float})\s+({re_float})\s+({re_float})\s+({re_float})|tau[cd]_([a-zA-Z]+)\d+\s+({re_float})\s+({re_float})\s+({re_float})\s+({re_float})')
        class_pattern = re.compile(
            r'(DIRECT) COORDINATES|(CARTESIAN) COORDINATES')
        unit_pattern = re.compile(rf'UNIT = ({re_float}) Bohr')
        # for '|', it will match all the patterns which results in '' or None
        coord_class = list(class_pattern.search(contents).groups())
        remove_empty(coord_class)
        self.coord_class = coord_class[0]
        positions = []
        for data in pos_pattern.findall(contents):
            site = list(map(list, site_pattern.findall(data[0])))
            list(map(remove_empty, site))
            self.labels, pos = str_to_sites(site)
            positions.append(pos)
        positions = np.array(positions)
        if self.coord_class == 'CARTESIAN':
            unit = float(unit_pattern.search(contents).group(1))*BOHR_TO_A
            self.positions = positions*unit
        elif self.coord_class == 'DIRECT':
            self.positions = positions

        # Fermi energy
        fermi_pattern = re.compile(rf'EFERMI\s*=\s*({re_float})\s*eV')
        self._fermi = float(fermi_pattern.search(contents).group(1))

        # orbital
        atom_pattern = re.compile(r'READING ATOM TYPE\s*\d+([\s\S]+?)\n\n')
        label_pattern = re.compile(r'atom label\s*=\s*(\w+)')
        zeta_pattern = re.compile(r'L=\d+, number of zeta\s*=\s*(\d+)')
        self.zetas = dict()
        len_zetas = []
        for data in atom_pattern.findall(contents):
            label = label_pattern.search(data).group(1)
            self.zetas[label] = list(map(int, zeta_pattern.findall(data)))
            len_zetas.append(len(self.zetas[label]))
        self._max_L = np.max(len_zetas)
        self._max_M = 2*self._max_L+1

def remove_empty(a: list):
    """Remove '' and [] in `a`"""
    while '' in a:
        a.remove('')
    while [] in a:
        a.remove([])
    while None in a:
        a.remove(None)


def handle_data(data):
    data.remove('')

    def handle_elem(elem):
        elist = elem.split(' ')
        remove_empty(elist)  # `list` will be modified in function
        return elist
    return list(map(handle_elem, data))


if __name__ == '__main__':
    running_file = r'C:\Users\YY.Ji\Desktop\running_scf.log'
    pdos_file = r'C:\Users\YY.Ji\Desktop\PDOS'
    obj = ABACUSDOSParser(pdos_file, running_file)
    print(obj._max_L)
