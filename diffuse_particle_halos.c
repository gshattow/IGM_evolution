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
#define MODEL "W11"
#define unbound_file_name "unbound"
#define Rvir_dist "False"
#define FA_dist "True"

// check box_side
// check tot_particles=X^3
// replace ALL type, e.g diffuse
// replace ALL SIM

int nbins = 100;
int particles = 0;
int galaxies = 0;
int ub_particles = 0;
float box_side = 62.5;
int tot_particles = 270;
float min_stars = 0.1;

int fnr;
int first = 0;
int last = 7;
int lines = 0;

int tot_n_gals = 0;
int tot_n_trees = 0;
int onepercent;

int snapnum;
int numsnaps = 64;


int were_unbound[100000000] = {0};
int now_unbound[100000000] = {1};
//int radial_distance_unbound[100000][11] = {0};
//int radial_distance_bound[100000][11] = {0};
//int aperture_distance_unbound[100000][11] = {0};
//int aperture_distance_bound[100000][11] = {0};
float max_dist = 10.;
float r_FA_max = 5.; //Mpc/h

void bin_particles();
int read_particles();
void count_galaxies();
int read_galaxies();
int calculate_radial_distance();
int calculate_aperture_distance();
char outfl[200];
void outfile();
void write_galaxies();
void timestamp();
void write_radial_unbound(int gg, int radial_distance_unbound[]);
void write_radial_bound(int gg, int radial_distance_bound[]);
void write_aperture_unbound(int gg, int aperture_distance_unbound[]);
void write_aperture_bound(int gg, int aperture_distance_bound[]);
float physical(float distance);

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
static struct particle_data P[20000000];

int snaps[2] = {32, 63};

int main()
{
	timestamp();
	int ss = 0;
	for (ss = 0; ss < 1; ss++)
	{
		snapnum = snaps[ss];

		//	int ub_ids = unbound_ids();
		printf("Starting snapshot %d (z = %.2f)\t", snapnum, z[snapnum]);
		timestamp();
		ub_particles = read_unbound();
		timestamp();
		particles = read_particles();
		timestamp();
		galaxies = read_galaxies();
		timestamp();
		if (Rvir_dist =="True") 
		{
			calculate_radial_distance();
			timestamp();
		}
		if (FA_dist =="True") 
		{
			calculate_aperture_distance();
			timestamp();
		}
		write_galaxies();
		timestamp();
// 		if (Rvir_dist == "True") 
// 		{
// 			write_distances_unbound();
// 			timestamp();
// 			write_distances_bound();
// 			timestamp();
// 		}
// 		if (FA_dist == "True") 
// 		{
// 			write_aperture_unbound();
// 			timestamp();
// 			write_aperture_bound();
// 			timestamp();
// 		}
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


	onepercent = galaxies/100;


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


//			if (now_unbound[dumID] == 1)
			{
				P[dumID].Pos[0] = xpos;				
				P[dumID].Pos[1] = ypos;				
				P[dumID].Pos[2] = zpos;				

// 				P[pp].Vel[0] = xvel;				
// 				P[pp].Vel[1] = yvel;				
// 				P[pp].Vel[2] = zvel;	
				
				pp ++;			

			}
		}
	}
	fclose(fp);
	
	printf("%d particles total.\n", pp);
	return pp;

}

int calculate_radial_distance()
{
	printf("Calculating particle distances for %d particles around %d galaxies...\n", particles, galaxies);
	int pp, gg;	
	float xdiff, ydiff, zdiff;
	for (gg = 0; gg < galaxies; gg ++)
	{
		float Rvir = G[gg].Rvir;
		int radial_distance_unbound[(int)max_dist];
		int radial_distance_bound[(int)max_dist];
		int rr = 0;
		for (rr = 0; rr < (int)max_dist; rr ++)
		{
			radial_distance_unbound[rr] = 0;
			radial_distance_bound[rr] = 0;
			
		}

		for (pp = 0; pp < particles; pp ++)
		{
			xdiff = fabs(P[pp].Pos[0] - G[gg].Pos[0]);
			if (xdiff > box_side/2.) {xdiff = xdiff - box_side; }
			xdiff = physical(xdiff);
		

			if (xdiff < Rvir*max_dist)
			{
				ydiff = fabs(P[pp].Pos[1] - G[gg].Pos[1]);
				if (ydiff > box_side/2.) {ydiff = ydiff - box_side; }
				ydiff = physical(ydiff);

				if (ydiff < Rvir*max_dist)
				{
					zdiff = fabs(P[pp].Pos[2] - G[gg].Pos[2]);
					if (zdiff > box_side/2.) {zdiff = zdiff - box_side; }
					zdiff = physical(zdiff);
				
					if (zdiff < Rvir*max_dist)
					{
						float dist = sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff);
						int radial_index = (int)(dist/Rvir);
						if (radial_index < 10 && now_unbound[pp] == 1) 
						{
							radial_distance_unbound[radial_index] ++; 
						}
						if (radial_index < 10 && now_unbound[pp] == 0) 
						{
							radial_distance_bound[radial_index] ++;
						}
				
					}				
			
				}
			
			}

		}


		if (gg % onepercent == 0) 
		{
			printf("%d percent done (%d galaxies), %d particles within 3Rvir.\t", 
				(gg/onepercent), gg, radial_distance_unbound[0] + radial_distance_unbound[1] +
					radial_distance_unbound[2] + radial_distance_unbound[3]);
			timestamp();
		}

		write_radial_unbound(gg, radial_distance_unbound);
		write_radial_bound(gg, radial_distance_bound);

	}


	return particles;

}

int calculate_aperture_distance()
{
	printf("Calculating particle distances for %d particles around %d galaxies...\n", particles, galaxies);
	int pp, gg;	
	float xdiff, ydiff, zdiff;
	int rbins = (int)r_FA_max*2;
	for (gg = 0; gg < galaxies; gg ++)
	{
		int aperture_distance_unbound[rbins];
		int aperture_distance_bound[rbins];
//		printf("%f, %d\n", r_FA_max*2, (int)r_FA_max*2);

		int rr = 0;
		for (rr = 0; rr < rbins; rr ++)
		{
			aperture_distance_unbound[rr] = 0;
			aperture_distance_bound[rr] = 0;
			
		}

		for (pp = 0; pp < particles; pp ++)
		{
			xdiff = fabs(P[pp].Pos[0] - G[gg].Pos[0]);
			if (xdiff > box_side/2.) {xdiff = xdiff - box_side; }
			xdiff = physical(xdiff);

			if (xdiff < r_FA_max)
			{
				ydiff = fabs(P[pp].Pos[1] - G[gg].Pos[1]);
				if (ydiff > box_side/2.) {ydiff = ydiff - box_side; }
				ydiff = physical(ydiff);

				if (ydiff < r_FA_max)
				{
					zdiff = fabs(P[pp].Pos[2] - G[gg].Pos[2]);
					if (zdiff > box_side/2.) {zdiff = zdiff - box_side; }
					zdiff = physical(zdiff);
				
					if (zdiff < r_FA_max)
					{
						float dist = sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff);
						int radial_index = (int)(dist/r_FA_max*rbins);
						if (radial_index < 10 && now_unbound[pp] == 1) 
						{
							aperture_distance_unbound[radial_index] ++; 
						}
						if (radial_index < 10 && now_unbound[pp] == 0) 
						{
							aperture_distance_bound[radial_index] ++;
						}
				
					}				
			
				}
			
			}

		}

		if (gg % onepercent == 0) 
		{
			printf("%d percent done (%d galaxies), %d particles within 2 Mpc/h.\t", 
				(gg/onepercent), gg, aperture_distance_unbound[0] + aperture_distance_unbound[1] +
					aperture_distance_unbound[2] + aperture_distance_unbound[3]);
			timestamp();
		}

		write_aperture_unbound(gg, aperture_distance_unbound);
		write_aperture_bound(gg, aperture_distance_bound);



	}


	return particles;

}

void write_galaxies()
{
	char output_fname[200];
	sprintf(output_fname, "%s%s%s_%03d", directory, grid_dir, "halos_109_comoving", snapnum);


	FILE *fq;
	fq = fopen(output_fname, "w");
	if (fq == NULL) { printf("Can't open outfile %s!\n", output_fname); }
	int gg = 0;
	
	for (gg = 0; gg < galaxies; gg ++)
	{
		fprintf(fq, "%d\t %lld\t %f\t %f\t %f\t", 
			G[gg].Type, G[gg].GalaxyIndex, G[gg].CentralMvir, G[gg].Mvir, G[gg].Rvir);
	
		fprintf(fq, "%f\t %f\t %f\t %f\t", 
			G[gg].ColdGas, G[gg].StellarMass, G[gg].HotGas, G[gg].EjectedMass);
			
		int ii = 0;
		for (ii = 0; ii < 3; ii ++)
		{
			fprintf(fq, "%f\t", G[gg].Pos[ii]);
		}
		for (ii = 0; ii < 3; ii ++)
		{
			fprintf(fq, "%f\t", G[gg].Vel[ii]);
		}
		fprintf(fq, "\n");
	
	}
	printf("%d galaxies printed to file %s.\n", gg, output_fname);
	
	fclose(fq);

}

void write_radial_unbound(int gg, int radial_distance_unbound[])
{
	char output_fname[200];
	sprintf(output_fname, "%s%s%s_%03d", directory, grid_dir, "unbound_particles_around_halos_Rvir", snapnum);


	FILE *fq;
	if (gg == 0)
	{
		fq = fopen(output_fname, "w");
	}
	else
	{
		fq = fopen(output_fname, "a");
	}

	if (fq == NULL) { printf("Can't open outfile %s!\n", output_fname); }
//		int gg = 0;

	
//	for (gg = 0; gg < galaxies; gg ++)
	int rr = 0;
	for (rr = 0; rr < max_dist; rr ++)
	{
		fprintf(fq, "%d\t", radial_distance_unbound[rr]);
		radial_distance_unbound[rr] = 0;
	}
	fprintf(fq, "\n");
//	}
//	printf("%d galaxies printed to file %s.\n", gg, output_fname);
	
	fclose(fq);

}

void write_radial_bound(int gg, int radial_distance_bound[])
{
	char output_fname[200];
	sprintf(output_fname, "%s%s%s_%03d", directory, grid_dir, "bound_particles_around_halos_Rvir", snapnum);


	FILE *fq;
	if (gg == 0)
	{
		fq = fopen(output_fname, "w");
	}
	else
	{
		fq = fopen(output_fname, "a");
	}
	if (fq == NULL) { printf("Can't open outfile %s!\n", output_fname); }
	
//	for (gg = 0; gg < galaxies; gg ++)
//	{
	int rr = 0;
	for (rr = 0; rr < max_dist; rr ++)
	{
		fprintf(fq, "%d\t", radial_distance_bound[rr]);
		radial_distance_bound[rr] = 0;
	}
	fprintf(fq, "\n");
	
//	}
//	printf("%d galaxies printed to file %s.\n", gg, output_fname);
	
	fclose(fq);

}

void write_aperture_unbound(int gg, int aperture_distance_unbound[])
{
	char output_fname[200];
	sprintf(output_fname, "%s%s%s_%03d", directory, grid_dir, "unbound_particles_around_halos_FA", snapnum);


	FILE *fq;
	if (gg == 0)
	{
		fq = fopen(output_fname, "w");
	}
	else
	{
		fq = fopen(output_fname, "a");
	}
	if (fq == NULL) { printf("Can't open outfile %s!\n", output_fname); }
//	int gg = 0;
	
//	for (gg = 0; gg < galaxies; gg ++)
//	{
	int rr = 0;
	for (rr = 0; rr < max_dist; rr ++)
	{
		fprintf(fq, "%d\t", aperture_distance_unbound[rr]);
		aperture_distance_unbound[rr] = 0;
	}
	fprintf(fq, "\n");
	
//	}
//	printf("%d galaxies printed to file %s.\n", gg, output_fname);
	
	fclose(fq);

}

void write_aperture_bound(int gg, int aperture_distance_bound[])
{
	char output_fname[200];
	sprintf(output_fname, "%s%s%s_%03d", directory, grid_dir, "bound_particles_around_halos_FA", snapnum);


	FILE *fq;
	if (gg == 0)
	{
		fq = fopen(output_fname, "w");
	}
	else
	{
		fq = fopen(output_fname, "a");
	}
	if (fq == NULL) { printf("Can't open outfile %s!\n", output_fname); }
//	int gg = 0;
	
//	for (gg = 0; gg < galaxies; gg ++)
//	{
	int rr = 0;
	for (rr = 0; rr < max_dist; rr ++)
	{
		fprintf(fq, "%d\t", aperture_distance_bound[rr]);
		aperture_distance_bound[rr] = 0;
	}
	fprintf(fq, "\n");
	
//	}
//	printf("%d galaxies printed to file %s.\n", gg, output_fname);
	
	fclose(fq);

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
