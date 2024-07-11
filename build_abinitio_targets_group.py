import pathlib
import sys
import os
import pickle
from collections import defaultdict
from pathlib import Path
import copy
import numpy as np
import qcportal
from forcebalance.molecule import Molecule as FBMolecule
from forcebalance.nifty import au2kcal, fqcgmx, au2kj
from openeye import oechem
from openff.qcsubmit.results import OptimizationResultCollection
from openff.toolkit.topology import Molecule
from qcelemental import periodictable
from qcelemental.molutil import guess_connectivity
from simtk import unit
from openff.toolkit.utils import UndefinedStereochemistryError
from tempfile import NamedTemporaryFile
import click
from openff.qcsubmit.results.filters import (
    SMARTSFilter,
    SMILESFilter,
)

# Script to set up ab initio targets
# Change parameters in targets_input.write around line 345 to change target parameters e.g. weights

# A lot of this code is borrowed from Pavan

client = qcportal.FractalClient()#.from_file()


# Group together conformers that correspond to the same molecule, and check 
# atom ordering, connectivity, and isomorphism.
def group_and_check_molecules(records_and_molecules): # modify reference_dict in place
    reference_dict = {} # smiles: Molecule--reference for remapping
    group_dict = {} # smiles: [(Molecule,remap)]
    remove_records = []
    n_errs = 0
    n_nonisomorph = 0

    # Iterate over the molecules
    for i, value in enumerate(records_and_molecules): # value is a (record,molecule) tuples

        record, molecule = list(value)#[0] # This just picks first conformer
        smiles=molecule.to_smiles(explicit_hydrogens=True,isomeric=False) # If it's unmapped, should be the same for all conformers (?)

        # reference_mols,grouped_mols = group_and_check_molecules(molecule,record,key,reference_mols,grouped_mols)

        # First instance--check connectivity, then add to dict, set reference
        if smiles not in reference_dict.keys():
            # print('First time seeing ',smiles)
            expected_connectivity = {
                tuple(sorted([bond.atom1_index, bond.atom2_index]))
                for bond in molecule.bonds
            }
            qc_molecule = molecule.to_qcschema()
            actual_connectivity = {
                tuple(sorted(connection))
                for connection in guess_connectivity(
                    qc_molecule.symbols, qc_molecule.geometry, 1.2
                )
            }
            if expected_connectivity != actual_connectivity:
                print("Connectivity check failed for ", smiles, ' QCAID ',record.id)
                n_errors += 1
                continue
            else:

                # molecule_remapped = molecule
                reference_dict[smiles] = (record,molecule)
                # get remap dict for completeness
                truth_value, remapping = Molecule.are_isomorphic(molecule,
                                                                 molecule,
                                                                 return_atom_map=True)

                group_dict[smiles] = [(record,molecule,remapping)]
        # Molecule is already in dict--check connectivity and remapping
        else:
            # print('Found existing molecule ',smiles)
            # Check connectivity (not sure this is necessary?)
            expected_connectivity = {
                tuple(sorted([bond.atom1_index, bond.atom2_index]))
                for bond in molecule.bonds
            }
            qc_molecule = molecule.to_qcschema()
            actual_connectivity = {
                tuple(sorted(connection))
                for connection in guess_connectivity(
                    qc_molecule.symbols, qc_molecule.geometry, 1.2
                )
            }
            if expected_connectivity != actual_connectivity:
                print("Connectivity check failed for ", smiles, ' QCAID ',record.id)
                n_errors += 1
                continue
            else:
                # Isomorphism check-- should we keep the remapping?
                truth_value, remapping = Molecule.are_isomorphic(molecule,
                                                                 reference_dict[
                                                                     smiles][1],
                                                                 return_atom_map=True)

                if truth_value:
                    group_dict[smiles].append((record,molecule,remapping))

                else:
                    print("failed isomorphic check", smiles, ' QCAID ',record.id)
                    n_nonisomorph += 1
                    remove_records.append(i)

    print('Number of records:',len(records_and_molecules))
    print('Number of connectivity errors: ',n_errs)
    print('Number of isomorphism fails: ',n_nonisomorph)
    return reference_dict,group_dict,remove_records


# Remap the gradient of a conformer according to a remapping dictionary
def remap_grads_geoms(grad,remap):
    # grad is np array Natomx3
    # remap is a dict of {old:new,old:new...}
    Natoms = grad.shape[0]
    assert Natoms == len(remap.keys()) # make sure the number of atoms are the same

    remap_idx_list= np.zeros(Natoms,dtype=int) # This will swap rows. if it turns out the atoms are columns, will have to transpose later
    for source_idx in remap.keys():
        dest_idx = remap[source_idx]

        # numpy will take this and pull the element at index i and put it in the new place
        # e.g. [1,0,3,2] will transform to [array[1],array[0],array[3],array[2]]
        # our dict is source: destination so we want remap= [array[dest=0],array[dest=1]...]--> have to swap the order
        remap_idx_list[dest_idx] = source_idx

    # print(remap_idx_list)
    return grad[remap_idx_list] # check that this is rows

# Get gradient from QCA record
def get_grad_from_record(item,molecule_from_record):
    if item.extras["qcvars"] == None:
        print(1)
        conf_geom = np.array(
            item.get_molecule().geometry) * \
                    unit.bohr.conversion_factor_to(unit.angstrom) * \
                    unit.angstrom

        # if true_val and remapping == None:
            # print(2)
        grad = np.array(
            item.return_result).reshape(item.properties.calcinfo_natom,
                                        3)
    elif "SCF TOTAL GRADIENT" in item.extras["qcvars"].keys():
        # print(5)
        conf_geom = np.array(
            item.get_molecule().geometry) * \
                    unit.bohr.conversion_factor_to(unit.angstrom) * \
                    unit.angstrom

        grad = np.array(
            item.extras["qcvars"]["SCF TOTAL GRADIENT"]).reshape(
            item.properties.calcinfo_natom, 3)  # in atomic units

    elif "-D GRADIENT" in item.extras["qcvars"].keys():
        # print(9)
        conf_geom = np.array(
            item.get_molecule().geometry) * \
                    unit.bohr.conversion_factor_to(unit.angstrom) * \
                    unit.angstrom

        grad = np.array(item.extras["qcvars"]["-D GRADIENT"]).reshape(
            item.properties.calcinfo_natom, 3)  # in atomic units

    return conf_geom,grad


# Turn data into FB targets
def create_targets(records_and_molecules, target_path):

    reference_molecules,grouped_molecules,remove_records = group_and_check_molecules(records_and_molecules)

    # # Create targets directory and write the input files
    Path(target_path).mkdir(parents=True, exist_ok=True)
    targets_input = open(target_path+'/og_ai_targets.in', 'w')
    errored_out_records = open(target_path+'/errored_out_targets.out', 'w')
    targets_excluded = open(target_path + '/targets_with_single_conformer_excluded.txt', 'w')

    for i, (smiles,value) in enumerate(grouped_molecules.items()): # value will be (molecule,record,remap)
        key = smiles # in case for legacy code

        record,molecule,remap = list(value)[0] # This just picks first conformer? or not?
        # print(value) # list of molecules in this group (record,molecule,remap)
        # print(list(value)[0]) # first conformer
        # print(record,molecule,remap) # self explan

        # Check whether there are at least two conformers to take energy difference
        if len(value) < 2:
            print("Single molecule, need at least two conformers for relative energy, ignoring this record", key)
            targets_excluded.write(f"QCA Record ID:{record.id}, SMILES: {molecule.to_smiles(mapped=True)}\n")
            continue
    #
        # Create target directory
        # Note that this isn't really the QCAID--should change that? or not?
        target_dir = target_path + '/targets/Abinitio_QCA-' + str(
            i) + '-' + molecule.hill_formula
        name = os.path.basename(target_dir)

        # Extract energies and gradients from the optimization records
        energies = []
        record_ids = []

        for item in value:
            # record_ids.append(item[0].id)
            record_ids.append(item[0].trajectory[-1]) # This is the result record ID of the final molecule
            energies.append(item[0].get_final_energy())

        try:
            energies = np.array(energies)  # in atomic units
            # print(energies)
            queried_records = client.query_results(record_ids)
            # print(queried_records)

            # all the records for these conformers?
            # why do it this way?
            final_records = []
            for id in record_ids:
                for item in queried_records:
                    # print(item.id)
                    if item.id == id:
                        final_records.append(item)

            # print(final_records)
            record_ids = np.array(record_ids)
            gradients = []
            xyzs = []

            molecule_from_record = copy.deepcopy(molecule)
            no_match_final_records = []

            for ir, item in enumerate(final_records):
                remap_ir = value[ir][2]
                # print(remap_ir)
                # Create a new molecule, delete conformers
                if ir != 0:
                    # print(0)
                    molecule_from_record = Molecule.from_smiles(key,
                                                                allow_undefined_stereo=True)
                    molecule_from_record._conformers = None

                conf_geom,grad=get_grad_from_record(item,molecule_from_record)

                molecule_from_record_remap = molecule_from_record.remap(remap_ir) # remap based on this record's remap

                conf_geom_remap = remap_grads_geoms(conf_geom, remap_ir)
                # print(conf_geom_remap)
                molecule_from_record_remap.add_conformer(conf_geom_remap)

                grad_remap = remap_grads_geoms(grad,remap_ir)
                gradients.append(grad_remap)  # in atomic units
                xyzs.append(conf_geom_remap)

                # one_to_one = {}
                # for n in range(0,molecule_from_record.n_atoms):
                #     one_to_one[n] = n
                # if remap_ir != one_to_one:
                #     print(remap_ir)
                #     print(conf_geom)
                #     print(conf_geom_remap)
                #     print(grad)
                #     print(grad_remap)


            for popper in sorted(no_match_final_records, reverse=True):
                np.delete(energies, popper)
                np.delete(record_ids, popper)


            gradients = np.array(gradients)
            xyzs = np.array(xyzs)
            # print(len(energies),gradients.shape,xyzs.shape)

            if len(xyzs) == 0:
                errored_out_records.write(f"{i}, {key}\n")
                continue
            # Create target directory for this molecule Abinitio_QCA-ID-hill_formula
            #
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            # print('Created target directory')

            # Create a FB molecule object from the QCData molecule.
            fb_molecule = FBMolecule()
            fb_molecule.Data = {
                "resname": ["UNK"] * molecule.n_atoms,
                "resid": [0] * molecule.n_atoms,
                "elem": [atom.symbol for atom in molecule.atoms],
                "bonds": [
                    (bond.atom1_index, bond.atom2_index) for bond in molecule.bonds
                ],
                "name": ', '.join(record_ids[np.argsort(energies)]),
                "xyzs": list(xyzs[np.argsort(energies)]),
                "qm_grads": list(gradients[np.argsort(energies)]),
                "qm_energies": list(np.sort(energies)),
                "comms": [f'QCA ID: {id}, Energy = {ener} hartree' for id, ener in
                          zip(record_ids[np.argsort(energies)], np.sort(energies))]
            }
            coord_file_name = f'{target_dir}/mol-{str(i)}'
            # Write the data
            fb_molecule.write(target_dir + '/qdata.txt')
            fb_molecule.write(coord_file_name + '.xyz')

            # Write XYZ file in sorted energies order and sdf, pdb files of a single conformer for FB topologies
            sdf_file_output = f'{target_dir}/mol-{str(i)}.sdf'
            pdb_file_output = f'{target_dir}/mol-{str(i)}.pdb'

            # Write optgeo_options file within the energy level target directory
            fname = open(target_dir + '/optgeo_options.txt', 'w')
            # print('writing target files')

            fname.write("$global \n"
                        "  bond_denom 0.05\n"
                        "  angle_denom 5\n"
                        "  dihedral_denom 10\n"
                        "  improper_denom 10\n"
                        "$end \n")
            fname.write(f"$system \n"
                        f"  name {name}\n"
                        f"  geometry mol-{i}.pdb\n"
                        f"  topology mol-{i}.xyz\n"
                        f"  mol2 mol-{i}.sdf\n"
                        f"$end\n")

            molecule._conformers = list()
            molecule._properties["SMILES"] = molecule.to_smiles(mapped=True)
            molecule.add_conformer(fb_molecule.Data["xyzs"][0] * unit.angstrom)
            molecule.to_file(sdf_file_output, file_format='sdf')
            fb_molecule.Data["xyzs"] = [fb_molecule.Data["xyzs"][0]]
            del fb_molecule.Data["qm_energies"]
            del fb_molecule.Data["qm_grads"]
            del fb_molecule.Data["comms"]
            fb_molecule.write(pdb_file_output)

            # print('adding target to input')

            # Write targets to a file that can be used in optimize_debug.in
            targets_input.write(f"$target \n"
                                f"  name {name}\n"
                                f"  type AbInitio_SMIRNOFF\n"
                                f"  mol2 mol-{i}.sdf\n"
                                f"  pdb mol-{i}.pdb\n"
                                f"  coords mol-{i}.xyz\n"
                                f"  writelevel 2\n"
                                f"  energy 1\n"
                                f"  force 1\n"
                                f"  w_energy 1.0\n"
                                f"  w_force 0.01\n"
                                f"  fitatoms 0\n"  ## all atoms
                                f"  remote 1\n"
                                f"  energy_rms_override 1\n"
                                f"  force_rms_override 10.0\n"
                                f"  energy_mode qm_minimum\n"  ## 'average', 'qm_minimum', or 'absolute'
                                f"  energy_asymmetry 1.0\n"  ## Assign a greater weight to MM snapshots that underestimate the QM energy (surfaces referenced to QM absolute minimum)
                                f"  attenuate 1\n"
                                f"  energy_denom 1.0\n"
                                f"  energy_upper 8.0\n"
                                f"  openmm_platform Reference\n"
                                f"$end\n"
                                f"\n")
        except Exception as e:
            print(e)
            pass

if __name__ == '__main__':
    inputfile = sys.argv[1]
    targetpath = sys.argv[2]
    training_set = OptimizationResultCollection.parse_file(inputfile)
    # print(training_set.n_molecules,training_set.n_results)

    exclude_smarts = pathlib.Path('smarts-to-exclude.dat').read_text().splitlines()
    exclude_smiles = pathlib.Path('smiles-to-exclude.dat').read_text().splitlines()   


    optimization_training_set = training_set.filter(
        SMARTSFilter(smarts_to_exclude=exclude_smarts),
        SMILESFilter(smiles_to_exclude=exclude_smiles),
    )

    records_and_molecules = optimization_training_set.to_records()

    create_targets(records_and_molecules,targetpath)
