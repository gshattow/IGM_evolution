#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_integration.h>
#include <time.h>
#define directory "/home/gshattow/HALOS/"
#define pcl_dir "/projects/p004_swin/abibiano/genev/minimill/"
#define grid_dir "rvir_files/"
#define galaxy_dir "/home/gshattow/sage/output/results/"
#define file_name "snap"
#define unbound_file_name "unbound"
#define MODEL "W11"
#define ID_file "IDs_evolution"
#define mass_range "14"

// check box_side
// check tot_particles=X^3
// replace ALL type, e.g diffuse
// replace ALL SIM


int nwindows = 5;
int nbins = 100;
int particles = 0;
int galaxies = 0;
int ub_particles = 0;
int nIDs = 0;
float box_side = 62.5;
int tot_particles = 270;
float min_stars = 0.1;

int fnr;
int first = 0;
int last = 7;
int lines = 0;

int tot_n_gals = 0;
int tot_n_trees = 0;

int snapnum;
int numsnaps = 64;


int now_unbound[100000000] = {0};
long long IDs[1000] = {0};
float max_dist = 10.;


void bin_particles();
int read_particles();
void count_galaxies();
int read_galaxies();
int read_galaxies_of_interest();
int calculate_window();
int calculate_fixed_window();
char outfl[200];
void outfile();
void write_galaxies();
void timestamp();
void write_distances();

float xmin = 0.; //5.8;
float ymin = 0.; //2.5;
float zmin = 0.; //12.5;

struct galaxy_data
{
  int   Type;
  long long   GalaxyIndex;
  int   HaloIndex;
  int   FOFHaloIndex;
  int   TreeIndex;
  
  int   SnapNum;
  int   CentralGal;
  float CentralMvir;

  // properties of subhalo at the last time this galaxy was a central galaaxy 
  float Pos[3];
  float Vel[3];
  float Spin[3];
  int   Len;   
  float Mvir;
  float Rvir;
  float Vvir;
  float Vmax;
  float VelDisp;

  // baryonic reservoirs 
  float ColdGas;
  float StellarMass;
  float BulgeMass;
  float HotGas;
  float EjectedMass;
  float BlackHoleMass;
  float ICS;

  // metals
  float MetalsColdGas;
  float MetalsStellarMass;
  float MetalsBulgeMass;
  float MetalsHotGas;
  float MetalsEjectedMass;
  float MetalsICS;

  // to calculate magnitudes
  float SfrDisk;
  float SfrBulge;
  float SfrICS;
  
  // misc 
  float DiskScaleRadius;
  float Cooling;
  float Heating;
  float LastMajorMerger;
  float OutflowRate;

  //infall properties
  float infallMvir;
  float infallVvir;
  float infallVmax;
  float r_heat;
};

struct particle_data
{
  float Pos[3];
  float Vel[3];
  int bound;
};


float z[64] = {127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 11.897, 
	10.944, 10.073, 9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 
		3.308, 3.060, 2.831, 2.619, 2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 1.078, 
			0.989, 0.905, 0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 
				0.208, 0.175, 0.144, 0.116, 0.089, 0.064, 0.041, 0.020, 0.000};



static struct galaxy_data G[100000];
static struct particle_data P[20000000];



int main()
{
	timestamp();
	nIDs = read_galaxies_of_interest();
	int sn;
	int snaps[1] = {32}; //, 40, 47, 63};
	for (sn = 0; sn < 1; sn ++)
	{
		snapnum = snaps[sn];
		printf("Starting snapshot %d (z = %.2f)\n", snapnum); //, z[snapnum]);
		timestamp();
		galaxies = read_galaxies();
		timestamp();
		ub_particles = read_unbound();
		timestamp();
		particles = read_particles();
		timestamp();
		calculate_window();
		timestamp();
		calculate_fixed_window();
// 		write_galaxies();
// 		timestamp();
// 		write_distances();
		timestamp();

		printf("\n\n\n");
	}
	return 0;
}

int read_galaxies_of_interest()
{
		
	int ii;
	for (ii = 0; ii < 1000; ii ++) {IDs[ii] = 0 ;}	
	/* Open the file of data */
	char input_fname[200];
	sprintf(input_fname, "%s%s_%s.dat", grid_dir, ID_file, mass_range);
	printf("loading particle list... %s\n", input_fname);

	FILE *fp;
	fp = fopen(input_fname, "r");
	if (fp == NULL)
	{
		printf("Can't open file %s!\n", input_fname);
	}

	nIDs = 0;

	ii = 0;
	for ( ; ; )
	{
		if (feof(fp))
		{
			break;
		}
		else
		{
			long long dumID;
			fscanf(fp,"%llu\n", &dumID);
			IDs[ii] = dumID;
			ii ++;
			nIDs ++;
		}
	}
	fclose(fp);
	
	printf("%d IDs read.\n", nIDs);
	printf("first few galaxies:\n");
//	for (ii = 0; ii < 100; ii ++) {printf("%lld\t", IDs[ii]);}
	
	
	
	if (nIDs < nwindows) { nwindows = nIDs; }
	
	return nIDs;

}

int read_galaxies()
{

	int numgal = 0;

	
	int offset = 0;
	tot_n_gals = 0;
	tot_n_trees = 0;
	float zej = 0.;

	for (fnr = first; fnr < last + 1; fnr ++)
	{

		char infl[200];
		sprintf(infl, "%s/millennium/%s%0.3f_%d", galaxy_dir, "model_z", z[snapnum], fnr);

		printf(" %d\t", fnr);
			

		FILE *fp;
		fp = fopen(infl, "rb");
		if (fp == NULL)
		{
			printf("Can't open file!\n");
		}
// 		else
// 		{
// 			printf("Now reading in galaxies from file: %s.\n", infl);
// 		}
	
		int numtrees;
		int numfile;
		fread(&numtrees, sizeof(int), 1, fp);
		fread(&numfile, sizeof(int), 1, fp);
		int galspertree[numtrees];
		fread(&galspertree, sizeof(int), numtrees, fp);
		int len = numfile;
		tot_n_trees += numtrees;
		tot_n_gals += numfile;
		
		if (numfile == 0) {printf("File %d has no galaxies!\n", fnr);}
		else {printf("File %d has %d galaxies!\n", fnr, len);}


		struct galaxy_data *GG = (struct galaxy_data *) malloc(len * sizeof( struct galaxy_data ));	
		if (GG == NULL) { printf("Uh Oh...\n"); }

		fread(GG, sizeof(struct galaxy_data), len, fp);

		fclose(fp);
		
		int gg = 0;
		int hh = 0;
		int ii = 0;
		int file_off = 0;

		for (hh = 0; hh < len; hh ++)
		{
			for (ii = 0; ii < nIDs; ii ++)
			{
				if (GG[hh].GalaxyIndex == IDs[ii])
				{
					IDs[ii] = -1;
					G[offset + gg] = GG[hh];		
					file_off ++;
					galaxies ++;
					gg ++;
				}
			}

		}
		free(GG);
		offset += file_off;
		
		printf("File %d has %d galaxies (%d total) of interest.\n", fnr, gg, offset);

	}

	lines = offset;


	printf("\n%d galaxies in ID list out of %d galaxies in %d trees total.\n", offset, tot_n_gals, tot_n_trees);
	if (nIDs != offset) {printf("Missing some galaxies, here!\n");}

	return lines;
	
	
}

int read_unbound()
{
		
	int ii;
	for (ii = 0; ii < 100000000; ii ++) {now_unbound[ii] = 0 ;}	
	/* Open the file of data */
	char input_fname[200];
//	sprintf(input_fname, "%s%s%s_%03d", directory, pcl_dir, "diffuse_particles", snapnum);
	sprintf(input_fname, "%s%s_%02d", pcl_dir, unbound_file_name, snapnum);
	printf("loading particle list... %s\n", input_fname);

	FILE *fp;
	fp = fopen(input_fname, "r");
	if (fp == NULL)
	{
		printf("Can't open file %s!\n", input_fname);
	}

	ub_particles = 0;
	int maxID = 0;
	int minID = 5000;
	float x_max = 0.;
	float y_max = 0.;
	float z_max = 0.;

	for ( ; ; )
	{
		if (feof(fp))
		{
			break;
		}
		else
		{
			int dumID;
			float xpos, ypos, zpos;
			fscanf(fp,"%d,%f,%f,%f\n", 
				&dumID, &xpos, &ypos, &zpos);

			now_unbound[dumID] = 1;
			if (dumID < minID) {minID = dumID;}
			if (dumID > maxID) {maxID = dumID;}
			if (xpos > x_max) {x_max = xpos;}
			if (ypos > y_max) {y_max = ypos;}
			if (zpos > z_max) {z_max = zpos;}
			ub_particles ++;
		}
	}
	fclose(fp);
	
	printf("minID, maxID, tot particles: %d, %d, %d.\n", 
		minID, maxID, tot_particles*tot_particles*tot_particles); 
	printf("%d unbound particles.\n", ub_particles); 
	printf("(xmax, ymax, zmax) = (%.2f, %.2f, %.2f)\n", x_max, y_max, z_max);
	
	return ub_particles;

}


int read_particles()
{
	int pp = 0;
		
	/* Open the file of data */
	char input_fname[200];
	sprintf(input_fname, "%s%s_%02d", pcl_dir, file_name, snapnum);
	printf("loading particle list... %s\n", input_fname);

	FILE *fp;
	fp = fopen(input_fname, "r");
	if (fp == NULL)
	{
		printf("Can't open file %s!\n", input_fname);
	}

	for ( ; ; )
	{
		if (feof(fp))
		{
			break;
		}
		else
		{
			int dumID, dumf;
			int xbin, ybin, zbin;
			float xpos, ypos, zpos;
			float xvel, yvel, zvel;
			fscanf(fp,"%d,%f,%f,%f,%f,%f,%f\n", 
				&dumID, &xpos, &ypos, &zpos, &xvel, &yvel, &zvel);


			P[pp].Pos[0] = xpos;				
			P[pp].Pos[1] = ypos;				
			P[pp].Pos[2] = zpos;				
			
			if (now_unbound[dumID] == 1)
			{
				P[pp].bound = -1;	
			}				
			else
			{
				P[pp].bound = 1;
			}

			pp ++;			


		}
	}
	fclose(fp);
	
	return pp;

}

int calculate_window()
{
	printf("Calculating particle distances for %d particles around %d galaxies...\n", particles, galaxies);
	int pp, gg;	
	float xdiff, ydiff, zdiff;
	for (gg = 0; gg < nwindows; gg ++)
	{
		int np = 0;
		char output_fname[200];
		sprintf(output_fname, "%s%s%s_%s_%03d_%03d", directory, grid_dir, "halo_window", 
			mass_range, gg, snapnum);
		printf("Galaxy %d: %lld\t %f\t %f\n", gg, G[gg].GalaxyIndex, G[gg].Mvir, G[gg].Rvir);


		FILE *fq;
		fq = fopen(output_fname, "w");
		if (fq == NULL) { printf("Can't open outfile %s!\n", output_fname); }

		fprintf(fq, "%lld\t %f\t %f\n", G[gg].GalaxyIndex, G[gg].Mvir, G[gg].Rvir);

		float Rvir8 = G[gg].Rvir*(1. + z[snapnum])*8.0;
//		printf("%d: 8 Rvir = %f Comoving Mpc/h \n", gg, Rvir8);
		for (pp = 0; pp < particles; pp ++)
		{
			xdiff = fabs(P[pp].Pos[0] - G[gg].Pos[0]);
			if (xdiff > box_side/2.) {xdiff = xdiff - box_side; }
			
			if (xdiff < Rvir8)
			{
				ydiff = fabs(P[pp].Pos[1] - G[gg].Pos[1]);
				if (ydiff > box_side/2.) {ydiff = ydiff - box_side; }

				if (ydiff < Rvir8)
				{
					zdiff = fabs(P[pp].Pos[2] - G[gg].Pos[2]);
					if (zdiff > box_side/2.) {zdiff = zdiff - box_side; }
					
					if (zdiff < Rvir8)
					{
						int ii = 0;
						for (ii = 0; ii < 3; ii ++)
						{
							float dist = P[pp].Pos[ii] - G[gg].Pos[ii];
							if (dist > box_side/2.) { dist -= box_side/2. ; }
							if (dist < -box_side/2.) { dist += box_side/2. ; }
							fprintf(fq, "%f\t", dist);
						}
						fprintf(fq, "%d\n", P[pp].bound);
						np ++;
					}				
				
				}
				
			}
			
		}
		fclose(fq);



	}


	return particles;

}

int calculate_fixed_window()
{
	printf("Calculating particle distances for %d particles around %d galaxies...\n", particles, galaxies);
	int pp, gg;	
	float xdiff, ydiff, zdiff;
	for (gg = 0; gg < nwindows; gg ++)
	{
		int np = 0;
		char output_fname[200];
		sprintf(output_fname, "%s%s%s_%s_%03d_%03d", directory, grid_dir, "halo_fixed_window", 
			mass_range, gg, snapnum);
		printf("Galaxy %d: %lld\t %f\t %f\n", gg, G[gg].GalaxyIndex, G[gg].Mvir, G[gg].Rvir);


		FILE *fq;
		fq = fopen(output_fname, "w");
		if (fq == NULL) { printf("Can't open outfile %s!\n", output_fname); }

		fprintf(fq, "%lld\t %f\t %f\n", G[gg].GalaxyIndex, G[gg].Mvir, G[gg].Rvir);

		float rad = 2.0;
//		printf("%d: 8 Rvir = %f Comoving Mpc/h \n", gg, Rvir8);
		for (pp = 0; pp < particles; pp ++)
		{
			xdiff = fabs(P[pp].Pos[0] - G[gg].Pos[0]);
			if (xdiff > box_side/2.) {xdiff = xdiff - box_side; }
			
			if (xdiff < rad)
			{
				ydiff = fabs(P[pp].Pos[1] - G[gg].Pos[1]);
				if (ydiff > box_side/2.) {ydiff = ydiff - box_side; }

				if (ydiff < rad)
				{
					zdiff = fabs(P[pp].Pos[2] - G[gg].Pos[2]);
					if (zdiff > box_side/2.) {zdiff = zdiff - box_side; }
					
					if (zdiff < rad)
					{
						int ii = 0;
						for (ii = 0; ii < 3; ii ++)
						{
							float dist = P[pp].Pos[ii] - G[gg].Pos[ii];
							if (dist > box_side/2.) { dist -= box_side/2. ; }
							if (dist < -box_side/2.) { dist += box_side/2. ; }
							fprintf(fq, "%f\t", dist);
						}
						fprintf(fq, "%d\n", P[pp].bound);
						np ++;
					}				
				}
			}
		}
		fclose(fq);
	}
	return particles;
}


void timestamp()
{
    time_t ltime; /* calendar time */
    ltime=time(NULL); /* get current cal time */
    printf("%s",asctime( localtime(&ltime) ) );
}
