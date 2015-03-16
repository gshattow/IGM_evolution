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
#define space "physical"

// check box_side
// check tot_particles=X^3
// replace ALL type, e.g diffuse
// replace ALL SIM

int nbins = 100;
int galaxies = 0;
float box_side = 62.5;
float min_stars = 0.1;

int fnr;
int first = 0;
int last = 7;
int lines = 0;

int tot_n_gals = 0;
int tot_n_trees = 0;

int snapnum;
int numsnaps = 64;

float r_FA_max = 5.;


void count_galaxies();
int read_galaxies();
int calculate_fixed_aperture();
void write_N_environment();
void write_rhobar_environment();
void timestamp();
float sphere(float radius);
float physical(float radius);

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
};


float z[64] = {127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 11.897, 
	10.944, 10.073, 9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 
		3.308, 3.060, 2.831, 2.619, 2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 1.078, 
			0.989, 0.905, 0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 0.408, 0.362, 0.320, 0.280, 0.242, 
				0.208, 0.175, 0.144, 0.116, 0.089, 0.064, 0.041, 0.020, 0.000};


static struct galaxy_data G[100000];
static int N_FA[100000][11];


int main()
{
	timestamp();
	for (snapnum = 32; snapnum < numsnaps; snapnum++)
	{
//		int ub_ids = unbound_ids();
		printf("Starting snapshot %d (z = %.2f)\t", snapnum, z[snapnum]);
		timestamp();
		galaxies = read_galaxies();
		timestamp();
		calculate_fixed_aperture();
		timestamp();
		write_N_environment();
		timestamp();
//		write_rhobar_environment();
//		timestamp();
		printf("\n\n\n");
	}
	return 0;
}

void count_galaxies()
{
	tot_n_gals = 0;
	tot_n_trees = 0;

	for (fnr = first; fnr < last + 1; fnr ++)
	{
		char infl[200];
		sprintf(infl, "millennium_%s/%s%0.3f_%d", MODEL, "model_z", z[snapnum], fnr);
		
		/* Open the file of data */
		FILE *fp;
		fp = fopen(infl, "rb");
		if (fp == NULL)
		{
			printf("Can't open file %s!\n", infl);
		}

		int numtrees;
		int numfile;
		fread(&numtrees, sizeof(int), 1, fp);
		fread(&numfile, sizeof(int), 1, fp);
		int galspertree[numtrees];
		fread(&galspertree, sizeof(int), numtrees, fp);

		tot_n_trees += numtrees;
		tot_n_gals += numfile;

		if (numfile == 0) {printf("File %d has no galaxies!\n", fnr);}


		fclose(fp);
		
	}
	printf("Total of %d galaxies found in %d trees.\n", tot_n_gals, tot_n_trees);

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
		int file_off = 0;

		for (hh = 0; hh < len; hh ++)
		{
			if (GG[hh].StellarMass > min_stars)
			{
				G[offset + gg] = GG[hh];		
				file_off ++;
				galaxies ++;
				gg ++;
			}
		}

		free(GG);
		offset += file_off;
		
		printf("%d\t", offset);


	}

	int lines = offset;


	printf("\n%d galaxies in mass range out of %d galaxies in %d trees total.\n", offset, tot_n_gals, tot_n_trees);
	printf("Galaxy 14: (Stars, CGas, Hotgas) = (%.2f, %.2f, %.2f)\n", 
		G[14].StellarMass, G[14].ColdGas, G[14].HotGas);
	int ii = 0;
	for (ii = 0; ii < 20; ii ++) printf("%.2f\t", G[ii].StellarMass);

	return lines;
	
	
}

int calculate_fixed_aperture()
{
	float r_FA_s = r_FA_max;//*(1. + z[snapnum]);
//	if (space == "comoving") {r_FA_s = r_FA;}
//	if (space == "physical") {r_FA_s = physical(r_FA);}

	printf("Calculating %.0f Mpc apertures around %d galaxies...\n", r_FA_max, galaxies);
	int hh, gg;	
	float xdiff, ydiff, zdiff;
	for (gg = 0; gg < galaxies; gg ++)
	{
//		printf("%d\n", gg);
		for (hh = 0; hh < galaxies; hh ++)
		{
			xdiff = fabs(G[hh].Pos[0] - G[gg].Pos[0]);
			if (xdiff > box_side/2.) {xdiff = xdiff - box_side; }
			if (space == "physical") {xdiff = physical(xdiff);}
			
			if (xdiff < r_FA_s)
			{
				ydiff = fabs(G[hh].Pos[1] - G[gg].Pos[1]);
				if (ydiff > box_side/2.) {ydiff = ydiff - box_side; }
				if (space == "physical") {ydiff = physical(ydiff);}

				if (ydiff < r_FA_s)
				{
					zdiff = fabs(G[hh].Pos[2] - G[gg].Pos[2]);
					if (zdiff > box_side/2.) {zdiff = zdiff - box_side; }
					if (space == "physical") {zdiff = physical(zdiff);}

					if (zdiff < r_FA_s)
					{
						float dist = sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff);
						
						float r_i = dist/r_FA_s*10.;
						int radial_index = (int)r_i;
						if (radial_index < 10) { N_FA[gg][radial_index] ++; }
					
					}				
				
				}
				
			}
			
		}

	}


	return gg;

}

void write_N_environment()
{
	char output_fname[200];
	sprintf(output_fname, "%s%s%s_%s_%03d", directory, grid_dir, "FA_N", space, snapnum);


	FILE *fq;
	fq = fopen(output_fname, "w");
	if (fq == NULL) { printf("Can't open outfile %s!\n", output_fname); }
	int gg = 0;
	
	for (gg = 0; gg < galaxies; gg ++)
	{
//		fprintf(fq, "%lld\t", G[gg].GalaxyIndex);
		
		int ii = 0;
		for (ii = 0; ii < 10; ii ++)
		{
			fprintf(fq, "%d\t", N_FA[gg][ii]);
			N_FA[gg][ii] = 0;
		}
		fprintf(fq, "\n");
//		N_FA[gg] = 0;
	
	}
	
	printf("%d galaxies printed to file %s.\n", gg, output_fname);
	
	fclose(fq);

}

void write_rhobar_environment()
{
	char output_fname[200];
	sprintf(output_fname, "%s%s%s_%s_%03d", directory, grid_dir, "FA_rho", space, snapnum);


	float N_average[10];
	int ii = 0;
	for (ii = 0; ii < 10; ii ++)
	{
		float radius = r_FA_max/10.*(float)(ii + 1);
		float volume = sphere(radius);
		N_average[ii] = galaxies/pow(box_side, 3)*volume;
		printf("%.2f\t, %.2f\n", volume, N_average[ii]);
	}

	FILE *fq;
	fq = fopen(output_fname, "w");
	if (fq == NULL) { printf("Can't open outfile %s!\n", output_fname); }
	int gg = 0;
	
	for (gg = 0; gg < galaxies; gg ++)
	{
		fprintf(fq, "%lld\t", G[gg].GalaxyIndex);
		
		int ii = 0;
		for (ii = 0; ii < 10; ii ++)
		{
			fprintf(fq, "%f\t", N_FA[gg][ii]/N_average[ii]);


//			printf("%f\t", N_FA[gg][ii]/N_average[ii]);			


			N_FA[gg][ii] = 0;
		}
		fprintf(fq, "\n");
	
	}
	
	printf("%d galaxies printed to file %s.\n", gg, output_fname);
	
	fclose(fq);

}

float sphere(float radius)
{
	if (space == "physical") {radius = physical(radius);}

	float volume = 4./3.*3.14159*pow(radius, 3);
	return volume;

}

float physical(float distance)
{
	distance = distance/(1 + z[snapnum]);

	return distance;

}

void timestamp()
{
    time_t ltime; /* calendar time */
    ltime=time(NULL); /* get current cal time */
    printf("%s",asctime( localtime(&ltime) ) );
}
