__author__ = "Yuyang Ji"
__maintainer__ = "Yuyang Ji"
__email__ = "jiyuyang@mail.ustc.edu.cn"
__date__ = "July 22, 2022"


import os
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict
import numpy as np
from scipy.constants import physical_constants

from pyprocar.core import DensityOfStates, Structure, ElectronicBandStructure, KPath

BOHR_TO_A = physical_constants["atomic unit of length"][0] / \
    physical_constants["Angstrom star"][0]

class ABACUSParser:
    def __init__(self, pdos_file='PDOS', pband_file='PBANDS_1', k_file='KLINES', running_file='running_scf.log', dos_interpolation_factor=None):

        self.dos_interpolation_factor = dos_interpolation_factor

        self.read(pdos_file, pband_file, k_file, running_file)

        if self.nspin in [1, 4]:
            self.is_spin_polarized = False
        elif self.nspin == 2:
            self.is_spin_polarized = True

        self.orbital_name = [
            's',
            'p_x', 'p_y', 'p_z',
            'd_xy', 'd_zy', 'd_z^2', 'd_zx', 'd_x^2-y^2',
            'f_z^3', 'f_xz^2', 'f_yz^2', 'f_xyz', 'f_z(x^2-y^2)',
            'f_x(x^2-3y^2)', 'f_y(3x^2-y^2)',
            'g_z^4', 'g_xz^3', 'g_yz^3', 'g_xyz^2', 'g_z^2(x^2-y^2)',
            'g_x^3z', 'g_y^3z', 'g_x^4+y^4', 'g_xy(x^2-y^2)',
        ]

## READ FILES

    def read(self, pdos_file, pband_file, k_file, running_file):
        if isinstance(pband_file, str):
            pband_file = [pband_file]
        self.pband_file = []
        for file in pband_file:
            if self.check_file(file):
                self.pband_file.append(file)
        self.pdos_file = self.check_file(pdos_file)
        self.k_file = self.check_file(k_file)
        self.running_file = self.check_file(running_file)

        if self.running_file:
            self._readABACUSout()
        if self.k_file:
            self._readKLINES()
        if self.pdos_file:
            self._readPDOS()
        if self.pband_file:
            self._readPBANDS()

    @staticmethod
    def check_file(filename):
        if os.path.isfile(filename):
            return filename
        else:
            return None

    def _readPDOS(self):
        """Read partial DOS data file"""

        self.dos_data = dict()
        pdosdata = ET.parse(self.pdos_file)
        root = pdosdata.getroot()
        self.nspin = int(root.findall('.//nspin')[0].text.replace(' ', ''))
        self._nsplit = self.nspin if self.nspin == 2 else 1
        norbitals = int(root.findall('.//norbitals')[0].text.replace(' ', ''))
        self.dos_data['eunit'] = root.findall(
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
        self.dos_data['energies'] = np.array(e_list).astype(float)
        self.dos_data['state'] = state

    def _readPBANDS(self):
        """Read partial DOS data file"""

        self.band_data = []
        for file in self.pband_file:
            band = dict()
            pbanddata = ET.parse(file)
            root = pbanddata.getroot()
            self.nspin = int(root.findall('.//nspin')[0].text.replace(' ', ''))
            self._nsplit = self.nspin if self.nspin == 2 else 1
            assert len(self.pband_file) == self._nsplit, f'{self._nsplit} projected band files should be read, only {len(self.pband_file)} provided here'
            norbitals = int(root.findall('.//norbitals')[0].text.replace(' ', ''))
            band['eunit'] = root.findall(
            './/band_structure')[0].attrib['units'].replace(' ', '')
            band['nbands'] = root.findall(
            './/band_structure')[0].attrib['nbands'].replace(' ', '')
            band['nkpoints'] = root.findall(
            './/band_structure')[0].attrib['nkpoints'].replace(' ', '')
            energy = root.findall('.//band_structure')[0].text.split('\n')
            energy = handle_data(energy)
            remove_empty(energy)
            band['band_structure'] = np.array(energy, dtype=float)
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
            band['state'] = state
            self.band_data.append(band)

    def _readKLINES(self):
        """Read KLINES file"""

        re_float = r'[\d\.\-\+]+'

        with open(self.k_file, 'r') as fd:
            contents = fd.read()
        label_pattern = re.compile(r'K_POINTS|KPOINTS|K')
        label = label_pattern.search(contents).group()
        number_pattern = re.compile(rf'{label}\s*\n(\d+)\s*\nLine')
        number = number_pattern.search(contents)
        if number:
            number = int(number.group(1))
        else:
            raise ValueError('File {self.k_file} with wrong settings')

        lines_pattern = re.compile(rf'({re_float})\s*({re_float})\s*({re_float})\s*(\d+)')
        lines = list(map(list, lines_pattern.findall(contents)))
        kpoints = np.array(lines)[:,:3].astype(float)[:number]
        self.ngrids = np.array(lines)[:,3].astype(int)[:number]
        self.kticks = (np.cumsum(np.concatenate(([1], self.ngrids)))-1)[:number]

        name_pattern = re.compile(rf'\d+\s*#(\w+)')
        name = name_pattern.findall(contents)[:number]
        assert len(name) == len(kpoints), 'Now we only support Line mode with high symmetry points.'

        self.special_kpoints = np.zeros(shape=(len(self.kticks)-1, 2, 3))
        self.knames = []
        for itick in range(len(self.kticks)):
            if itick != len(self.kticks) - 1: 
                self.special_kpoints[itick,0,:] = kpoints[itick]
                self.special_kpoints[itick,1,:] = kpoints[itick+1]
                self.knames.append([name[itick], name[itick+1] ])

        has_time_reversal = True
        self.kpath = KPath(
                        knames=self.knames,
                        special_kpoints=self.special_kpoints,
                        kticks = self.kticks,
                        ngrids=self.ngrids,
                        has_time_reversal=has_time_reversal,
                    )

    def _readABACUSout(self):
        """Read running_*.log file"""

        re_float = r'[\d\.\-\+Ee]+'

        def str_to_sites(val_in):
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

        def str_to_kpoints(val_in):
            lines = re.search(
                rf'KPOINTS\s*DIRECT_X\s*DIRECT_Y\s*DIRECT_Z\s*WEIGHT([\s\S]+?)DONE', val_in).group(1).strip().split('\n')
            data = []
            for line in lines:
                data.append(line.strip().split()[1:5])
            kpoints, weights, _ = np.split(
                np.array(data, dtype=float), [3, 4], axis=1)
            return kpoints, weights.flatten()

        with open(self.running_file, 'r') as fd:
            contents = fd.read()

        self.nspin = int(re.search(r'nspin\s*=\s*(\d+)', contents).group(1))
        self._nsplit = self.nspin if self.nspin == 2 else 1

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
        self._M_list = []
        for hl in len_zetas:
            m = []
            for l in range(hl):
                m.append(2*l+1)
            self._M_list.append(m)
        self._max_M_list = list(map(np.sum, self.M_list))
        self._max_M = np.max(self._max_M_list)

        # kpoints
        k_pattern = re.compile(r'minimum distributed K point number\s*=\s*\d+([\s\S]+?DONE : INIT K-POINTS Time)')
        sub_contents = k_pattern.search(contents).group(1)
        self.kpoints, self.weigths = str_to_kpoints(sub_contents)

# STRUCTURE

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
    def reciprocal_lattice(self):
        """reciprocal_lattice of the last step"""
        return self.structure.reciprocal_lattice

## BAND STRUCTURE

    @property
    def fermi(self):
        """
        Returns the fermi energy read from running_*.log
        """

        return self._fermi

## DOS

    def _get_dos_total(self):

        dos_total = {'energies': self.dos_data['energies']}
        res = np.zeros_like(self.dos_data['state'][0]["data"], dtype=float)
        for orb in self.dos_data['state']:
            res = res + orb['data']

        if self.is_spin_polarized:
            dos_total['Spin-up'] = res[:, 0]
            dos_total['Spin-down'] = res[:, 1]
        else:
            dos_total['Spin-up'] = res[:, 0]

        return dos_total, list(dos_total.keys())

    def _get_dos_projected(self):
        """dos_projected[name][l_index][m_index].shape == (obj._nsplit, len(obj.dos_data['energies']))"""

        dos_projected = dict()
        res = np.zeros((len(self.dos_data['energies']), self._nsplit))
        for orb in self.dos_data['state']:
            elem = orb['species']
            name = elem+str(orb['atom_index']-1)
            l_index = orb['l']
            m_index = orb['m']
            z_index = orb['z']
            zs = self.zetas[elem][l_index]
            if z_index <= zs:
                res += orb['data']
            if z_index == zs:
                dos_projected[name] = {'energies': self.dos_data['energies']}
                dos_projected[name][l_index] = {m_index: res.T}
                res = np.zeros((len(self.dos_data['energies']), self._nsplit))

        return dos_projected, ['energies']+self.orbital_name[:self._max_M]

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
        
        dos_projected, info = self._get_dos_projected()
        if dos_projected is None:
            return None
        ret = np.zeros((self.initial_structure.natoms, self._max_L+1, self._max_M, self._nsplit, len(self.dos_data['energies'])), dtype=float)
        #for i_atom, atom_key in enumerate(dos_projected):
            # for i_principal in range(self._max_L):
            #     for i_orbital in range(self._max_M):
            #         if i_principal not in dos_projected[atom_key].keys() or i_orbital not in dos_projected[atom_key][i_principal].keys():
            #             ret[i_atom][i_principal][i_orbital] = np.zeros((self._nsplit, len(self.dos_data['energies'])), dtype=float)
            #         else:
            #             ret[i_atom][i_principal][i_orbital] = dos_projected[atom_key][i_principal][i_orbital]
            # ret[i_atom][self._max_L][0] = self.dos_total
        
        return ret


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
    pbands_file = ''
    k_file = r'C:\Users\YY.Ji\Desktop\KPT'
    obj = ABACUSParser(pdos_file, pbands_file, k_file, running_file)
    print(obj.max_M_list)