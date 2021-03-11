from __future__ import print_function

import mdtraj as md
import numpy as np
import argparse
import os


def load_traj(top_fn, traj_fn):
	traj = md.load(traj_fn, top=top_fn)
	return traj

def save_traj(traj, output_file='aa_mmgbsa.nc', start_frame=900, end_frame=1000):
	traj[start_frame:end_frame].save_netcdf(output_file)

def generate_MMGBSA_file():
	MMGBSA = '''Input file for running GB
&general
  startframe = 1,
  interval = 1,
  verbose = 2,
  keep_files = 2,
  endframe = 50,
/
&gb
  igb = 5,
  saltcon = 0.100,
/'''
	with open('MMGBSA.md', 'w') as f:
		f.writelines(MMGBSA)

def calc_MMGBSA(Cmp_top, Cmp_Vac_top, Rec_Vac_top, Lig_Vac_top, traj_MMGBSA, output_file):
	if not os.path.exists('MMGBSA.md'):
		generate_MMGBSA_file()

	command = 'MMPBSA.py -O -i MMGBSA.md -o %s -sp %s -cp %s -rp %s -lp %s -y %s' % (output_file, Cmp_top, Cmp_Vac_top, Rec_Vac_top, Lig_Vac_top, traj_MMGBSA)
	# MMPBSA.py -O -i MMGBSA.md -o MMGBSA.mdout -sp Cmp.prmtop -cp Cmp_Vac.prmtop -rp Rec_Vac.prmtop -lp Lig_Vac.prmtop -y aa_mmgbsa.nc
	os.system(command)

def get_parser():
	parser = argparse.ArgumentParser(description='Calculate mmbgsa with amber')
	parser.add_argument('-p', default='Cmp.prmtop', help='input topology file')
	parser.add_argument('-x', default='aa_solv.nc', help='input trajectory file')
	parser.add_argument('-s', default='50', type=int, help='start frame')
	parser.add_argument('-e', default='100', type=int, help='end frame')
	parser.add_argument('-o', default='aa_mmgbsa.nc', help='output file')
	parser.add_argument('-cp', default='Cmp_Vac.prmtop', help='Cmp vac topology')
	parser.add_argument('-rp', default='Rec_Vac.prmtop', help='Rec vac topology')
	parser.add_argument('-lp', default='Lig_Vac.prmtop', help='Lig vac topology')
	parser.add_argument('-y', default='aa_mmgbsa.nc', help='traj for mmgbsa')
	parser.add_argument('-mo', default='MMGBSA.mdout', help='output for mmgbsa')
	return parser.parse_args()

def scanning_path(current_path='./'):
	for _, folder, _ in os.walk(current_path):
		break
	return folder

def main():
	args = get_parser()
	in_top = args.p
	in_x = args.x
	in_s = args.s
	in_e = args.e
	outputfile = args.o
	traj = load_traj(in_top, in_x)
	print('load traj...')
	print('save traj...')
	save_traj(traj, outputfile, in_s, in_e)
	print('done...')

	cp = args.cp
	rp = args.rp
	lp = args.lp
	mo = args.mo
	mmpbsa_traj = args.y
	#MMPBSA.py -O -i MMGBSA.md -o MMGBSA.mdout -sp Cmp.prmtop -cp Cmp_Vac.prmtop -rp Rec_Vac.prmtop -lp Lig_Vac.prmtop -y aa_900to1000.nc
	print('Calculate MMGBSA')
	calc_MMGBSA(in_top, cp, rp, lp, mmpbsa_traj, mo)

if __name__ == '__main__':
	main()
	#python calc_MMGBSA.py -p Cmp.prmtop -x aa_solv.nc
