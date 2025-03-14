# 2-SFS

## Daniel P. Rice, Eliot F. Fenton, John Novembre, and Michael M. Desai

### Installation

#### Conda environment

```bash
conda env create -f config/environment.yml && conda activate twosfs && python -m pip install -e .
```

#### Git

Use git lfs for simulations.tgz

#### Instructions for running your own data through the pipeline:
1. Create the sitefile - for creation from vcf, see vcf_to_sites.sh. Save it in a directory, e.g.
   ./path/to/data/sites.json
2. Create param file - see example_param_file.json. You will need to change the sample size,
   ploidy, and anything else you want to adjust. Note that ploidy only describes the ploidy of 
   the samples, not the species - e.g. a sample size of 100 where we have both chromosomes for
   each individual should have "num_samps" = 100 and "ploidy" = 2. All simulations are run as
   diploid. Save the param file as ./path/to/data/params.json
3. Edit snakefiles/data.snake, and create a new rule - e.g.:
      rule: my_data
          input:
              ["path/to/data/ks_distance.folded=True.json"],
          resources:
              time=5,
              mem=100,
   Note you can also use the unfolded SFS and 2-SFS by changing the above to folded=False
4. Run snakemake, e.g.: snakemake my_data. For deployment on a Slurm cluster, use:
   snakemake my_data --cluster 'sbatch -c 1 -t {resources.time} --mem={resources.mem}' -j 10
   The final argument can be increased to allow more jobs to run at once, see snakemake
   documentation for more details.
5. KS distances and p-values will be found at path/to/data/ks_distance.folded=True.json

