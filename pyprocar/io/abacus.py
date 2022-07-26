__author__ = "Yuyang Ji"
__maintainer__ = "Yuyang Ji"
__email__ = "jiyuyang@mail.ustc.edu.cn"
__date__ = "July 22, 2022"


import os
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
import numpy as np
from scipy.constants import physical_constants

from pyprocar.core import DensityOfStates, Structure, ElectronicBandStructure, KPath

BOHR_TO_A = physical_constants["atomic unit of length"][0] / \
    physical_constants["Angstrom star"][0]

class ABACUSParser:
    def __init__(self, pdos_file='PDOS', pband_file='PBANDS_1', k_file='KLINES', scf_log='running_scf.log', nscf_log='running_nscf.log', dos_interpolation_factor=None):

        self.dos_interpolation_factor = dos_interpolation_factor

        self.read(pdos_file, pband_file, k_file, scf_log, nscf_log)

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

    def orbitals(self, L):
        l_list = [i for i in range(L)]
        m_list = [[j for j in range(2*i+1)] for i in range(L)]

        return dict(zip(l_list, m_list))

    @property
    def projected_labels(self):
        projected_labels = dict()
        for i, label in enumerate(self.labels):
            L = len(self.zetas[label])
            projected_labels[i+1] = self.orbitals(L)

        return projected_labels


## READ FILES

    def read(self, pdos_file, pband_file, k_file, scf_log, nscf_log):
        if isinstance(pband_file, str):
            pband_file = [pband_file]
        self.pband_file = []
        for file in pband_file:
            if self.check_file(file):
                self.pband_file.append(file)
        self.pdos_file = self.check_file(pdos_file)
        self.k_file = self.check_file(k_file)
        self.scf_log = self.check_file(scf_log)
        self.nscf_log = self.check_file(nscf_log)

        if self.scf_log:
            self._readABACUSscfout()
        if self.nscf_log:
            self._readABACUSnscfout()
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
            self.nbands = int(root.findall(
            './/band_structure')[0].attrib['nbands'].replace(' ', ''))
            self.nkpoints = int(root.findall(
            './/band_structure')[0].attrib['nkpoints'].replace(' ', ''))
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

    def _readABACUSscfout(self):
        """Read running_scf.log file"""

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

        with open(self.scf_log, 'r') as fd:
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
        _M_list = []
        for hl in len_zetas:
            m = []
            for l in range(hl):
                m.append(2*l+1)
            _M_list.append(m)
        self._M_dict = dict(zip(self.zetas.keys(), _M_list))
        self._max_M_dict = dict(zip(self.zetas.keys(), list(map(np.cumsum, _M_list))))
        self._max_M = np.max(list(map(np.sum, _M_list)))

    def _readABACUSnscfout(self):
        """Read running_nscf.log file"""

        def str_to_kpoints(val_in):
            lines = re.search(
                rf'KPOINTS\s*DIRECT_X\s*DIRECT_Y\s*DIRECT_Z\s*WEIGHT([\s\S]+?)DONE', val_in).group(1).strip().split('\n')
            data = []
            for line in lines:
                data.append(line.strip().split()[1:5])
            kpoints, weights, _ = np.split(
                np.array(data, dtype=float), [3, 4], axis=1)
            return kpoints, weights.flatten()

        with open(self.nscf_log, 'r') as fd:
            contents = fd.read()

        # kpoints
        k_pattern = re.compile(r'minimum distributed K point number\s*=\s*\d+([\s\S]+?DONE : INIT K-POINTS Time)')
        sub_contents = k_pattern.search(contents).group(1)
        self.kpoints, self._kweights = str_to_kpoints(sub_contents)

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

    @property
    def bands(self):
        data = np.zeros(shape=(self.nkpoints, self.nbands, self._nsplit))
        for ispin, band in enumerate(self.band_data):
            data[:,:,ispin] = band['band_structure']

        return data

    def _get_weights(self):
        
        weights = []
        keyname = 'atom_index'
        for band in self.band_data:
            weight, totnum = parse_projected_data(band['state'], self.projected_labels, keyname)
            weights.append(weight)

        return weights, self.orbital_name[:self._max_M]

    @property
    def weight_atom_projected(self):
        keyname = 'atom_index'
        atom_index = np.array([i for i in range(self.initial_structure.natoms)])+1
        weights = []
        for band in self.band_data:
            weight, totnum = parse_projected_data(band['state'], atom_index, keyname)
            weights.append(weight)

        return weights

    @property
    def weights_projected(self):
        _weights, info = self._get_weights()
        if _weights is None:
            return None
        ret = np.zeros((self.nkpoints, self.nbands, self.initial_structure.natoms, self._max_L+1, self._max_M, self._nsplit))
        for ispin, weight in enumerate(_weights):
            for i_atom in weight.keys():
                atom_key = self.labels[i_atom-1]
                for l in range(len(self._M_dict[atom_key])):
                    for m in range(self._M_dict[atom_key][l]):
                        if l == 0:
                            ret[:, :, i_atom-1, l, m, ispin]= weight[i_atom][l][m]
                        else:                     
                            ret[:, :, i_atom-1, l, m+self._max_M_dict[atom_key][l-1], ispin] = weight[i_atom][l][m]
                ret[:, :, i_atom-1, -1, 0, ispin] = self.weight_atom_projected[ispin][i_atom]
        
        return ret

    @property
    def ebs(self):
        return ElectronicBandStructure(
            kpoints=self.kpoints,
            bands=self.bands,
            projected=self.weights_projected,
            efermi=self.fermi,
            kpath=self.kpath,
            labels=self.orbital_name,
            reciprocal_lattice=self.reciprocal_lattice,
            interpolation_factor=self.dos_interpolation_factor,
        )


## DOS

    def _get_dos_total(self):

        dos = {'energies': self.dos_data['energies']}
        res = np.zeros_like(self.dos_data['state'][0]["data"], dtype=float)
        for orb in self.dos_data['state']:
            res = res + orb['data']

        if self.is_spin_polarized:
            dos['Spin-up'] = res[:, 0]
            dos['Spin-down'] = res[:, 1]
        else:
            dos['Spin-up'] = res[:, 0]

        return dos, list(dos.keys())

    def _get_dos_projected(self):

        keyname = 'atom_index'
        dos_projected, totnum = parse_projected_data(self.dos_data['state'], self.projected_labels, keyname)
        #[nspin][ndos]
        return dos_projected, self.orbital_name[:self._max_M]

    @property
    def dos(self):
        energies = self.dos_data['energies'] - self.fermi
        return DensityOfStates(
            energies=energies,
            total=self.dos_total,
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

        dos, labels = self._get_dos_total()
        total = []
        for ispin in dos:
            if ispin == 'energies':
                continue
            total.append(dos[ispin]) # (nspin, ndos)

        return total

    @property
    def dos_atom_projected(self):
        keyname = 'atom_index'
        atom_index = np.array([i for i in range(self.initial_structure.natoms)])+1

        return parse_projected_data(self.dos_data['state'], atom_index, keyname)[0]

    @property
    def dos_projected(self):
        """
        Returns the projected DOS as a multi-dimentional array, to be used in the
        pyprocar.core.dos object
        """
        
        dos, info = self._get_dos_projected()
        if dos is None:
            return None
        ret = np.zeros((self.initial_structure.natoms, self._max_L+1, self._max_M, self._nsplit, len(self.dos_data['energies'])), dtype=float)
        for i_atom in dos.keys():
            atom_key = self.labels[i_atom-1]
            for l in range(len(self._M_dict[atom_key])):
                for m in range(self._M_dict[atom_key][l]):
                    if l == 0:
                        ret[i_atom-1][l][m] = dos[i_atom][l][m].T
                    else:
                        ret[i_atom-1][l][m+self._max_M_dict[atom_key][l-1]] = dos[i_atom][l][m].T
            ret[i_atom-1][-1][0] = self.dos_atom_projected[i_atom].T
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


def parse_projected_data(orbitals, species, keyname=''):
    """Extract projected data from file

    Args:
        species (Union[Sequence[Any], Dict[Any, List[int]], Dict[Any, Dict[str, List[int]]]], optional): list of atomic species(index or atom index) or dict of atomic species(index or atom index) and its angular momentum list. Defaults to [].
        keyname (str): the keyword that extracts the projected data. Allowed values: 'index', 'atom_index', 'species'
    """

    if isinstance(species, (list, tuple, np.ndarray)):
        data = {}
        elements = species
        for elem in elements:
            count = 0
            data_temp = np.zeros_like(orbitals[0]["data"], dtype=float)
            for orb in orbitals:
                if orb[keyname] == elem:
                    data_temp += orb["data"]
                    count += 1
            if count:
                data[elem] = data_temp

        return data, len(elements)

    elif isinstance(species, dict):
        data = defaultdict(dict)
        elements = list(species.keys())
        l = list(species.values())
        totnum = 0
        for i, elem in enumerate(elements):
            if isinstance(l[i], dict):
                for ang, mag in l[i].items():
                    l_count = 0
                    l_index = int(ang)
                    l_data = {}
                    for m_index in mag:
                        m_count = 0
                        data_temp = np.zeros_like(
                            orbitals[0]["data"], dtype=float)
                        for orb in orbitals:
                            if orb[keyname] == elem and orb["l"] == l_index and orb["m"] == m_index:
                                data_temp += orb["data"]
                                m_count += 1
                                l_count += 1
                        if m_count:
                            l_data[m_index] = data_temp
                            totnum += 1
                    if l_count:
                        data[elem][l_index] = l_data

            elif isinstance(l[i], list):
                for l_index in l[i]:
                    count = 0
                    data_temp = np.zeros_like(
                        orbitals[0]["data"], dtype=float)
                    for orb in orbitals:
                        if orb[keyname] == elem and orb["l"] == l_index:
                            data_temp += orb["data"]
                            count += 1
                    if count:
                        data[elem][l_index] = data_temp
                        totnum += 1

        return data, totnum


if __name__ == '__main__':
    running_file = r'C:\Users\YY.Ji\Desktop\running_scf.log'
    pdos_file = r'C:\Users\YY.Ji\Desktop\PDOS'
    pbands_file = r'C:\Users\YY.Ji\Desktop\PBANDS_1'
    k_file = r'C:\Users\YY.Ji\Desktop\KPT'
    obj = ABACUSParser(pdos_file, pbands_file, k_file, running_file)
    #print(obj.weights_projected)